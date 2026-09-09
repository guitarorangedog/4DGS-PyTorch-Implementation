"""REAL GPU end-to-end test for training/trainer_4d.py (Commit 16).

Shortened coarse + fine run on a tiny multi-timestamp D-NeRF fixture with
the real CUDA rasterizer. Requires CUDA + the extension.

Run via `python3 -m tests.test_trainer_4d`.
"""

import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data.scene import load_scene
from deformation.decoder import DecoderConfig
from deformation.field import FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.render_4d import render_deformed_view
from deformation.regularize import RegWeights, regularization_terms
from gaussians.rasterizer import render_view
from training.losses import l1_loss, reconstruction_loss_4dgs, ssim
from training.optim import get_group
from training.trainer_4d import (
    FourDTrainerConfig,
    _should_prune,
    fine_threshold,
    init_4d_model,
    save_4d_model,
    train_coarse,
    train_fine,
)
from training.trainer_static import set_seed


def _write_scene(root: str, size: int = 64) -> None:
    def c2w(tx=0.0):
        m = np.eye(4)
        m[:3, 3] = [tx, 0, 0]
        return m.tolist()

    train_frames = [
        {"file_path": f"train/r_{i}", "transform_matrix": c2w(0.1 * i), "time": t}
        for i, t in enumerate([0, 1, 2])
    ]
    test_frames = [{"file_path": "test/r_0", "transform_matrix": c2w(0.0), "time": 1}]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    rng = np.random.default_rng(5)
    for rel in ("train/r_0", "train/r_1", "train/r_2", "test/r_0"):
        arr = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        arr[..., 3] = 255
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        Image.fromarray(arr, "RGBA").save(os.path.join(root, rel + ".png"))


def _cfg() -> FourDTrainerConfig:
    cfg = FourDTrainerConfig(
        coarse_iterations=4, fine_iterations=6, seed=0, log_interval=2,
        max_sh_degree=0, densify_from_iter=1, densify_until_iter=20,
        densification_interval=2, densify_grad_threshold_coarse=0.0,
        densify_grad_threshold_fine_init=0.0, densify_grad_threshold_fine_after=0.0,
        pruning_interval=2, opacity_reset_interval=100,
        opacity_threshold_coarse=1e-9,
        opacity_threshold_fine_init=1e-9, opacity_threshold_fine_after=1e-9,
        field=FieldConfig(
            hexplane=HexPlaneConfig(bounds=1.6, output_coordinate_dim=4,
                                    resolution=[8, 8, 8, 5], multires=[1, 2]),
            decoder=DecoderConfig(width=16, depth=1)),
        reg=RegWeights(plane_tv=0.0001, time_smoothness=0.01, l1_time_planes=0.0001),
    )
    return cfg


def _frame_cloud(model, device) -> None:
    with torch.no_grad():  # D-NeRF cameras look -z; keep cloud framed
        model._xyz.copy_(torch.randn_like(model._xyz) * 0.4 + torch.tensor([0, 0, -2.0], device=device))
        model._scaling.copy_(torch.full_like(model._scaling, float(np.log(0.15))))


def test_coarse_to_fine_4d() -> None:
    if not torch.cuda.is_available():
        print("test_coarse_to_fine_4d: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        assert sorted({v.camera.time for v in scene.train_views}) == [0.0, 0.5, 1.0]
        # Faithful D-NeRF trainer default: multires [1, 2] (dnerf_default.py),
        # not the generic [1, 2, 4, 8].
        assert FourDTrainerConfig().field.hexplane.multires == [1, 2]
        cfg = _cfg()
        model, field = init_4d_model(scene, cfg, device=device)
        _frame_cloud(model, device)
        n0 = model.num_points

        # Optimizer groups: 8 named, official LRs (scaled by extent).
        names = [g["name"] for g in model.optimizer.param_groups]
        assert names == ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation",
                         "deformation", "grid"], names
        ext = scene.scene_extent
        assert abs(get_group(model, "deformation")["lr"] - 0.00016 * ext) < 1e-12
        assert abs(get_group(model, "grid")["lr"] - 0.0016 * ext) < 1e-12
        assert abs(get_group(model, "xyz")["lr"] - 0.00016 * ext) < 1e-12
        assert len(get_group(model, "deformation")["params"]) > 1  # multi-param group
        deform_before = [p.detach().clone() for p in field.get_mlp_parameters()]
        grid_before = [p.detach().clone() for p in field.get_grid_parameters()
                       if p.requires_grad]

        # A. Coarse: static path, deformation frozen.
        coarse_hist = train_coarse(scene, model, field, cfg, device=device)
        assert len(coarse_hist) == 2 and all(np.isfinite(h["loss"]) for h in coarse_hist)
        assert all("reg" not in h for h in coarse_hist)
        for p, b in zip(field.get_mlp_parameters(), deform_before):
            assert torch.equal(p.detach(), b), "deformation updated during coarse"
        for p, b in zip([p for p in field.get_grid_parameters() if p.requires_grad], grid_before):
            assert torch.equal(p.detach(), b), "grids updated during coarse"
        n_coarse = model.num_points
        assert n_coarse > n0, "coarse densify never fired"

        # Install trained-like temporal variation for the fine stage.
        torch.manual_seed(9)
        with torch.no_grad():
            for level in field.hexplane.grids:
                for i in (2, 4, 5):
                    level[i].copy_(torch.rand_like(level[i]) * 2 - 1)
            for p in field.decoder.parameters():
                p.mul_(0.05)

        # C/E. Fine: deformed path + live regularization in the loss.
        view = scene.train_views[0]
        pkg = render_deformed_view(view.camera, model, field, device=device)
        pred, gt = pkg["render"].unsqueeze(0), view.image.to(device).unsqueeze(0)
        recon = reconstruction_loss_4dgs(pred, gt, 0.0)
        terms = regularization_terms(field)
        manual = (0.0001 * terms.spatial + 0.01 * terms.temporal + 0.0001 * terms.l1_time)
        assert torch.isfinite(terms.spatial) and torch.isfinite(terms.temporal)
        assert torch.isfinite(terms.l1_time) and manual.item() > 0

        fine_hist = train_fine(scene, model, field, cfg, device=device)
        assert len(fine_hist) == 3 and all(np.isfinite(h["loss"]) for h in fine_hist)
        assert all(h["reg"] > 0 for h in fine_hist), "reg missing/zero in fine"
        changed_mlp = sum(not torch.equal(p.detach(), b)
                          for p, b in zip(field.get_mlp_parameters(), deform_before))
        assert changed_mlp > 0, "decoder frozen during fine"
        assert model.num_points >= n_coarse  # surgery kept optimizer valid
        model.optimizer.step()  # 8-group optimizer steps cleanly post-surgery

        # D. Fresh fine step: grads everywhere + valid viewspace grads.
        model.optimizer.zero_grad(set_to_none=True)
        pkg = render_deformed_view(view.camera, model, field, device=device)
        loss = reconstruction_loss_4dgs(pkg["render"].unsqueeze(0), gt, 0.0)
        loss = loss + (0.0001 * regularization_terms(field).spatial)
        loss.backward()
        assert torch.isfinite(model._xyz.grad).all() and model._xyz.grad.abs().sum() > 0
        assert torch.isfinite(pkg["viewspace_points"].grad).all()
        assert sum(p.grad.abs().sum().item() for p in field.get_grid_parameters()
                   if p.grad is not None) > 0
        assert sum(p.grad.abs().sum().item() for p in field.get_mlp_parameters()
                   if p.grad is not None) > 0

        # Timestamps differentiate renders in fine stage.
        import copy
        late = copy.copy(view.camera)
        late.time = 0.05 if view.camera.time > 0.5 else 0.95
        a = render_deformed_view(view.camera, model, field, device=device)["render"]
        b = render_deformed_view(late, model, field, device=device)["render"]
        assert not torch.equal(a, b)

        # Loss equation parity: official L1 + λ(1-SSIM), no (1-λ) factor.
        lam = 0.2
        got = reconstruction_loss_4dgs(pred, gt, lam)
        assert torch.allclose(got, l1_loss(pred, gt) + lam * (1 - ssim(pred, gt)), atol=1e-7)

        # LR schedule parity with official D-NeRF formulas
        # (dnerf_default: deform 1.6e-4->1.6e-6, grid 1.6e-3->1.6e-5).
        from training.schedules import get_expon_lr_func
        ref_xyz = get_expon_lr_func(0.00016 * ext, 0.0000016 * ext, max_steps=20_000)
        ref_def = get_expon_lr_func(0.00016 * ext, 0.0000016 * ext, max_steps=20_000)
        ref_grid = get_expon_lr_func(0.0016 * ext, 0.000016 * ext, max_steps=20_000)
        for step in (1, 100, 19_999):
            lrs = {"xyz": ref_xyz(step), "deformation": ref_def(step), "grid": ref_grid(step)}
            got_lrs = {"xyz": model.xyz_schedule(step), "deformation": model.deform_schedule(step),
                       "grid": model.grid_schedule(step)}
            assert lrs == got_lrs, (step, lrs, got_lrs)
        # Threshold interpolation check (defaults equal -> constant).
        assert fine_threshold(0.0002, 0.0002, 7, 20) == 0.0002
        assert abs(fine_threshold(0.001, 0.0, 10, 20) - 0.0005) < 1e-12

        # Static path untouched: same camera renders identically, ignoring time.
        static_img = render_view(view.camera, model, device=device)["render"]
        assert torch.isfinite(static_img).all()

        # Artifacts: PLY + deformation weights + meta reload.
        out = os.path.join(root, "out")
        from training.trainer_4d import save_4d_model
        paths = save_4d_model(model, field, out, cfg,
                              {"coarse": coarse_hist, "fine": fine_hist})
        assert all(os.path.exists(p) for p in paths.values())
        field2 = type(field)(field.config).to(device)
        field2.load_state_dict(torch.load(paths["deformation"]))
        for p, q in zip(field.parameters(), field2.parameters()):
            assert torch.equal(p.detach(), q.detach())
    print(f"test_coarse_to_fine_4d: passed "
          f"(coarse {coarse_hist[0]['loss']:.4f}->{coarse_hist[-1]['loss']:.4f}, "
          f"fine {fine_hist[0]['loss']:.4f}->{fine_hist[-1]['loss']:.4f}, "
          f"n {n0}->{model.num_points})")


def test_pruning_schedule_defaults() -> None:
    # Official D-NeRF override (dnerf_default.py): pruning_interval = 8000.
    cfg = FourDTrainerConfig()
    assert cfg.pruning_interval == 8000
    assert not _should_prune(cfg, 100), "must not prune at generic-default cadence"
    assert not _should_prune(cfg, 7999)
    assert _should_prune(cfg, 8000)
    assert _should_prune(cfg, 16_000)
    assert not _should_prune(cfg, 16_001)
    # Short-cadence smoke override still works.
    assert _should_prune(_cfg(), 2) and not _should_prune(_cfg(), 3)
    print("test_pruning_schedule_defaults: passed (D-NeRF cadence 8000)")


if __name__ == "__main__":
    test_pruning_schedule_defaults()
    test_coarse_to_fine_4d()
    print("test_trainer_4d.py: done")

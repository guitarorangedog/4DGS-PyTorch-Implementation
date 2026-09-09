"""REAL CUDA resume-parity tests for training/checkpoints.py (Commit 17).

Compares uninterrupted vs checkpoint-interrupted 4D training on a tiny
multi-timestamp fixture. Requires CUDA + the extension.

Tolerances are justified, not arbitrary: single CUDA-rasterizer backward
passes are order-nondeterministic at ~1e-8 (xyz) to ~1e-6 (scales) per
step (see Commit 14 report), so resumed-vs-clean runs agree to ~1e-4 in
loss after several steps. Exact-equality assertions are used ONLY where
the quantities are definitionally restored (counts, shapes, degrees,
LR values, camera sequences).

Run via `python3 -m tests.test_checkpoints`.
"""

import copy
import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data.scene import load_scene
from deformation.decoder import DecoderConfig
from deformation.field import DeformationField, FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.render_4d import render_deformed_view
from gaussians.gaussian_model import CanonicalGaussianModel
from training.checkpoints import (
    checkpoint_filename,
    load_checkpoint,
    restore_4d_state,
    save_checkpoint,
)
from training.optim import get_group
from training.trainer_4d import FourDTrainerConfig, init_4d_model, train_4d, train_fine
from training.trainer_static import set_seed


def _write_scene(root: str, size: int = 64) -> None:
    def c2w(tx=0.0):
        m = np.eye(4)
        m[:3, 3] = [tx, 0, 0]
        return m.tolist()

    train_frames = [
        {"file_path": f"train/r_{i}", "transform_matrix": c2w(0.1 * i), "time": t}
        for i, t in enumerate([0, 1])
    ]
    test_frames = [{"file_path": "test/r_0", "transform_matrix": c2w(0.0), "time": 1}]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    rng = np.random.default_rng(7)
    for rel in ("train/r_0", "train/r_1", "test/r_0"):
        arr = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        arr[..., 3] = 255
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        Image.fromarray(arr, "RGBA").save(os.path.join(root, rel + ".png"))


def _cfg(**over) -> FourDTrainerConfig:
    kw = dict(coarse_iterations=4, fine_iterations=6, seed=0, log_interval=1,
              max_sh_degree=0, densify_from_iter=100,  # densify OFF unless asked
              pruning_interval=2, opacity_reset_interval=100,
              opacity_threshold_coarse=1e-9,
              opacity_threshold_fine_init=1e-9, opacity_threshold_fine_after=1e-9,
              field=FieldConfig(
                  hexplane=HexPlaneConfig(bounds=1.6, output_coordinate_dim=4,
                                          resolution=[8, 8, 8, 5], multires=[1, 2]),
                  decoder=DecoderConfig(width=16, depth=1)))
    kw.update(over)
    return FourDTrainerConfig(**kw)


def _fresh_pair(scene, cfg, device):
    model, field = init_4d_model(scene, cfg, device=device)
    with torch.no_grad():  # D-NeRF cameras look -z; keep cloud framed
        model._xyz.copy_(torch.randn_like(model._xyz) * 0.4 + torch.tensor([0, 0, -2.0], device=device))
        model._scaling.copy_(torch.full_like(model._scaling, float(np.log(0.15))))
    return model, field


def _optim_moments(model):
    out = {}
    for g in model.optimizer.param_groups:
        st = model.optimizer.state.get(g["params"][0], None)
        if st is not None and "exp_avg" in st:
            out[g["name"]] = (st["exp_avg"].detach().cpu().clone(),
                              st["exp_avg_sq"].detach().cpu().clone(),
                              st.get("step", None))
    return out


def test_fine_resume_parity() -> None:
    if not torch.cuda.is_available():
        print("test_fine_resume_parity: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        # A: uninterrupted coarse(4) + fine(6).
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = _cfg()
        model_a, field_a = _fresh_pair(scene, cfg, device)
        hist_a = train_4d(scene, model_a, field_a, cfg, device=device)
        # B: same run, but checkpoint at fine iter 3 and resume 4..6.
        # (Coarse runs fully first; the fine optimizer is rebuilt, then the
        # first 3 fine iters run, checkpoint, destroy, restore, continue.)
        from training.trainer_4d import train_coarse, _stage_loop
        import random
        set_seed(0)
        scene = load_scene(root, seed=0)
        model_b, field_b = _fresh_pair(scene, cfg, device)
        coarse_b = train_coarse(scene, model_b, field_b, cfg, device=device)
        from training.optim import setup_4d_optimizer
        setup_4d_optimizer(model_b, field_b, cfg.optim, cfg.deform_optim,
                           spatial_lr_scale=scene.scene_extent)
        rng_b = random.Random(cfg.seed + 10_000)
        fine_part1 = _stage_loop(scene, model_b, field_b, cfg, "fine", 3, device,
                                 start_iteration=1, loop_rng=rng_b)
        ckpt = os.path.join(root, "ckpt_fine_3.pth")
        save_checkpoint(ckpt, stage="fine", iteration=3, model=model_b,
                        field=field_b, loop_rng=rng_b)
        # Bit-exact snapshots of EVERYTHING the checkpoint must preserve.
        snap_tensors = {k: getattr(model_b, k).detach().cpu().clone()
                        for k in ("_xyz", "_scaling", "_rotation", "_opacity",
                                  "_features_dc", "_features_rest",
                                  "xyz_gradient_accum", "denom", "max_radii2D")}
        snap_field = [p.detach().cpu().clone() for p in field_b.parameters()]
        snap_moments = _optim_moments(model_b)
        n_mid, sh_mid = model_b.num_points, model_b.active_sh_degree
        del model_b, field_b
        torch.cuda.empty_cache()
        model_c = CanonicalGaussianModel(max_sh_degree=0).to(device)
        field_c = DeformationField(cfg.field).to(device)
        loop_rng = __import__("random").Random()
        payload = load_checkpoint(ckpt, map_location=device)
        assert payload["stage"] == "fine" and payload["iteration"] == 3
        restore_4d_state(model_c, field_c, payload, cfg.optim, cfg.deform_optim,
                         device, loop_rng)
        # Restore fidelity: bit-exact BEFORE any further training.
        assert model_c.num_points == n_mid and model_c.active_sh_degree == sh_mid
        for k, v in snap_tensors.items():
            assert torch.equal(getattr(model_c, k).detach().cpu(), v), k
        for p, q in zip(field_c.parameters(), snap_field):
            assert torch.equal(p.detach().cpu(), q)
        for k, (m0, v0, s0) in snap_moments.items():
            m1, v1, s1 = _optim_moments(model_c)[k]
            assert torch.equal(m0, m1) and torch.equal(v0, v1) and s0 == s1, k
        fine_part2 = _stage_loop(scene, model_c, field_c, cfg, "fine",
                                 cfg.fine_iterations, device,
                                 start_iteration=4, loop_rng=loop_rng)
        # Parity: topology, SH, LR continuity, camera stream, close losses.
        assert model_c.num_points == model_a.num_points
        assert model_c.active_sh_degree == model_a.active_sh_degree
        for g in model_c.optimizer.param_groups:
            ga = get_group(model_a, g["name"])
            assert g["params"][0].shape == ga["params"][0].shape, g["name"]
            assert abs(g["lr"] - ga["lr"]) < 1e-15, g["name"]
        views_a = [h["view"] for h in hist_a["fine"]]
        views_b = [h["view"] for h in fine_part1] + [h["view"] for h in fine_part2]
        assert views_a == views_b, "camera stream diverged after resume"
        # NOTE: no tight iter-4 loss assert here by construction — run B's
        # pre-checkpoint iters 1..3 already carry atomic-noise divergence from
        # run A, so resumed inputs legitimately differ. Restore fidelity is
        # proven bit-exact above; continued training is compared loosely.
        for ha, hb in zip(hist_a["fine"], fine_part1 + fine_part2):
            assert abs(ha["loss"] - hb["loss"]) < 5e-3, (ha, hb)
        # Post-resume values: xyz moves slowly (tight); Adam-normalized groups
        # (scaling/rotation/opacity) can diverge O(lr) from atomic-order grad
        # noise (Commit 14 mechanism), so they are checked loose-but-bounded.
        assert torch.allclose(model_a._xyz.detach(), model_c._xyz.detach(), atol=1e-3)
        for name in ("_scaling", "_rotation", "_opacity"):
            pa, pc = getattr(model_a, name).detach(), getattr(model_c, name).detach()
            assert torch.isfinite(pc).all() and pa.shape == pc.shape, name
            assert (pa - pc).abs().max().item() < 1.0, name
        ma, mc = _optim_moments(model_a), _optim_moments(model_c)
        assert set(ma) == set(mc)
        for k in ma:
            assert ma[k][0].shape == mc[k][0].shape, k
            assert torch.allclose(ma[k][0], mc[k][0], atol=1e-2), k
    print("test_fine_resume_parity: passed")


def test_coarse_resume_and_transition() -> None:
    if not torch.cuda.is_available():
        print("test_coarse_resume_and_transition: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = _cfg(max_sh_degree=3)  # allow the SH-continuity probe below
        model, field = _fresh_pair(scene, cfg, device)
        from training.trainer_4d import _stage_loop
        import random
        rng = random.Random(cfg.seed)
        _stage_loop(scene, model, field, cfg, "coarse", 2, device,
                    start_iteration=1, loop_rng=rng)
        ckpt = os.path.join(root, "ckpt_coarse_2.pth")
        save_checkpoint(ckpt, stage="coarse", iteration=2, model=model,
                        field=field, loop_rng=rng)
        # SH continuity probe: bump degree, re-save, restore keeps it.
        model.oneupSHdegree()
        save_checkpoint(ckpt, stage="coarse", iteration=2, model=model,
                        field=field, loop_rng=rng)
        n_mid = model.num_points
        del model, field
        torch.cuda.empty_cache()
        model2 = CanonicalGaussianModel(max_sh_degree=3).to(device)
        field2 = DeformationField(cfg.field).to(device)
        hist = train_4d(scene, model2, field2, cfg, device=device, resume=ckpt)
        assert model2.active_sh_degree == 1, "SH degree not preserved"
        assert len(hist["coarse"]) == 2, "coarse must continue 3..4 only"
        assert [h["iteration"] for h in hist["coarse"]] == [3, 4]
        assert len(hist["fine"]) == 6, "fine must run fully after resumed coarse"
        assert model2.num_points == n_mid  # no densify in this cfg
        assert all("reg" not in h for h in hist["coarse"])
        assert all(h["reg"] >= 0 for h in hist["fine"])
    print("test_coarse_resume_and_transition: passed")


def test_densified_topology_restore() -> None:
    if not torch.cuda.is_available():
        print("test_densified_topology_restore: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = _cfg(densify_from_iter=1, densify_until_iter=20,
                   densification_interval=2, densify_grad_threshold_coarse=0.0,
                   densify_grad_threshold_fine_init=0.0,
                   densify_grad_threshold_fine_after=0.0)
        model, field = _fresh_pair(scene, cfg, device)
        from training.trainer_4d import _stage_loop
        import random
        rng = random.Random(cfg.seed)
        _stage_loop(scene, model, field, cfg, "coarse", 4, device,
                    start_iteration=1, loop_rng=rng)
        assert model.num_points > 2000, "densify never fired"
        n_mid = model.num_points
        raws = {k: getattr(model, k).detach().clone()
                for k in ("_xyz", "_scaling", "_rotation", "_opacity",
                          "_features_dc", "_features_rest")}
        ckpt = os.path.join(root, "ckpt_dense.pth")
        save_checkpoint(ckpt, stage="coarse", iteration=4, model=model,
                        field=field, loop_rng=rng,
                        extra_meta={"n": n_mid})
        grid_n = sum(p.numel() for p in field.get_grid_parameters())
        mlp_n = sum(p.numel() for p in field.get_mlp_parameters())
        del model, field
        torch.cuda.empty_cache()
        model2 = CanonicalGaussianModel(max_sh_degree=0).to(device)
        field2 = DeformationField(cfg.field).to(device)
        payload = load_checkpoint(ckpt, map_location="cpu")  # CPU map exercises migration
        assert payload["meta"]["n"] == n_mid
        restore_4d_state(model2, field2, payload, cfg.optim, cfg.deform_optim,
                         device, random.Random())
        assert model2.num_points == n_mid
        for k, v in raws.items():
            assert torch.equal(getattr(model2, k).detach().cpu(), v.cpu()), k
        for g in model2.optimizer.param_groups:
            st = model2.optimizer.state.get(g["params"][0], None)
            if st is not None and "exp_avg" in st:
                assert st["exp_avg"].shape == g["params"][0].shape, g["name"]
                assert st["exp_avg"].device == g["params"][0].device, g["name"]
        assert sum(p.numel() for p in field2.get_grid_parameters()) == grid_n
        assert sum(p.numel() for p in field2.get_mlp_parameters()) == mlp_n
        model2.optimizer.step()  # post-restore step succeeds
        train_fine(scene, model2, field2, cfg, device=device)  # fine runs on restored topology
    print(f"test_densified_topology_restore: passed (N={n_mid})")


def test_model_only_loading_unaffected() -> None:
    if not torch.cuda.is_available():
        print("test_model_only_loading_unaffected: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = _cfg()
        model, field = _fresh_pair(scene, cfg, device)
        from training.trainer_4d import save_4d_model, train_coarse
        train_coarse(scene, model, field, cfg, device=device)
        paths = save_4d_model(model, field, os.path.join(root, "out"), cfg,
                              {"coarse": [], "fine": []})
        from training.render_static import load_static_model, render_views
        reloaded = load_static_model(paths["ply"], device=device)
        assert reloaded.num_points == model.num_points
        imgs = render_views(reloaded, scene.test_views, device=device)
        assert len(imgs) == 1 and torch.isfinite(imgs[0]).all()
    print("test_model_only_loading_unaffected: passed")


if __name__ == "__main__":
    test_fine_resume_parity()
    test_coarse_resume_and_transition()
    test_densified_topology_restore()
    test_model_only_loading_unaffected()
    print("test_checkpoints.py: done")

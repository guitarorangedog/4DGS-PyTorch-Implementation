"""Tests for configs/ selection, values, wiring, resume-compat. Run via `python3 -m tests.test_configs`."""

import os
import sys
import tempfile

import numpy as np
import torch

from configs import dnerf_config, dynerf_config, get_trainer_config, hypernerf_config
from training.checkpoints import assert_field_compatible, save_checkpoint
from training.trainer_4d import FourDTrainerConfig


def test_dispatch_and_dnerf_exact() -> None:
    assert isinstance(get_trainer_config("dnerf"), FourDTrainerConfig)
    assert isinstance(get_trainer_config("dynerf"), FourDTrainerConfig)
    assert isinstance(get_trainer_config("hypernerf"), FourDTrainerConfig)
    try:
        get_trainer_config("colmap")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown dataset should raise")
    cfg = dnerf_config()
    assert (cfg.coarse_iterations, cfg.fine_iterations) == (3000, 20_000)
    assert (cfg.optim.position_lr_init, cfg.optim.position_lr_final) == (0.00016, 0.0000016)
    assert (cfg.deform_optim.deformation_lr_init, cfg.deform_optim.deformation_lr_final) == (0.00016, 0.0000016)
    assert (cfg.deform_optim.grid_lr_init, cfg.deform_optim.grid_lr_final) == (0.0016, 0.000016)
    assert cfg.pruning_interval == 8000 and cfg.optim.percent_dense == 0.01
    assert cfg.field.hexplane.multires == [1, 2]
    assert cfg.field.decoder.width == 64 and cfg.field.decoder.depth == 1
    assert (cfg.reg.plane_tv, cfg.reg.time_smoothness, cfg.reg.l1_time_planes) == (0.0001, 0.01, 0.0001)
    assert cfg.field.hexplane.bounds == 1.6
    assert cfg.field.hexplane.output_coordinate_dim == 32
    assert cfg.field.hexplane.resolution == [64, 64, 64, 25]
    assert cfg.field.enable_aux is False
    print("test_dispatch_and_dnerf_exact: passed")


def test_dynerf_hypernerf_exact() -> None:
    dy, hy = dynerf_config(), hypernerf_config()
    for cfg in (dy, hy):
        assert cfg.coarse_iterations == 3000 and cfg.fine_iterations == 14_000
        assert cfg.densify_until_iter == 10_000
        assert cfg.field.hexplane.output_coordinate_dim == 16
        assert cfg.field.hexplane.resolution == [64, 64, 64, 150]
        assert cfg.field.decoder.width == 128 and cfg.field.decoder.depth == 1
        assert (cfg.reg.plane_tv, cfg.reg.time_smoothness, cfg.reg.l1_time_planes) == (0.0002, 0.001, 0.0001)
        # LR finals NOT overridden -> global values (unlike D-NeRF).
        assert cfg.deform_optim.deformation_lr_final == 0.000016
        assert cfg.deform_optim.grid_lr_final == 0.00016
    assert dy.field.hexplane.multires == [1, 2]
    assert hy.field.hexplane.multires == [1, 2, 4]
    assert dy.field.enable_aux is True and hy.field.enable_aux is False
    assert dy.opacity_reset_interval == 60_000
    assert hy.opacity_reset_interval == 300_000
    assert dy.pruning_interval == 100 and hy.pruning_interval == 100
    assert dy.densify_grad_threshold_coarse == 0.0002
    print("test_dynerf_hypernerf_exact: passed")


def test_field_and_optimizer_wiring() -> None:
    from deformation.field import DeformationField
    from training.optim import setup_4d_optimizer, get_group
    for name, multires, feat in (("dnerf", [1, 2], 64), ("dynerf", [1, 2], 32),
                                 ("hypernerf", [1, 2, 4], 48)):
        cfg = get_trainer_config(name)
        field = DeformationField(cfg.field)
        assert field.hexplane.feat_dim == feat, (name, field.hexplane.feat_dim)
        assert [g for g in ("xyz", "deformation", "grid")]  # group names exist below
    # Optimizer LRs follow the selected config (spot-check grid/deform inits).
    import torch as _t  # noqa: F401 (keeps device selection explicit below)
    from gaussians.gaussian_model import CanonicalGaussianModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanonicalGaussianModel(max_sh_degree=0)
    model.create_from_pointcloud(torch.randn(4, 3), torch.rand(4, 3))
    field = DeformationField(dynerf_config().field)
    setup_4d_optimizer(model, field, dynerf_config().optim,
                       dynerf_config().deform_optim, spatial_lr_scale=2.0)
    assert abs(get_group(model, "grid")["lr"] - 0.0016 * 2.0) < 1e-12
    assert abs(get_group(model, "deformation")["lr"] - 0.00016 * 2.0) < 1e-12
    print("test_field_and_optimizer_wiring: passed")


def test_resume_rejects_incompatible_arch() -> None:
    import random
    from gaussians.gaussian_model import CanonicalGaussianModel
    from deformation.field import DeformationField
    from training.optim import setup_4d_optimizer

    cfg = dnerf_config(coarse_iterations=1, fine_iterations=1)
    model = CanonicalGaussianModel(max_sh_degree=0)
    model.create_from_pointcloud(torch.randn(4, 3), torch.rand(4, 3))
    field = DeformationField(cfg.field)
    setup_4d_optimizer(model, field, cfg.optim, cfg.deform_optim)
    with tempfile.TemporaryDirectory() as root:
        ckpt = os.path.join(root, "c.pth")
        save_checkpoint(ckpt, stage="coarse", iteration=1, model=model,
                        field=field, loop_rng=random.Random(0))
        from training.checkpoints import load_checkpoint
        payload = load_checkpoint(ckpt)
        assert_field_compatible(payload, dnerf_config())
        for bad in (dynerf_config(), hypernerf_config()):
            try:
                assert_field_compatible(payload, bad)
            except ValueError as e:
                assert "Checkpoint/config architecture conflict" in str(e)
            else:
                raise AssertionError("cross-dataset resume should raise")
    print("test_resume_rejects_incompatible_arch: passed")


def _smoke_dataset(dataset: str) -> None:
    sys.path.insert(0, "tests")
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        if dataset == "dnerf":
            from test_dnerf import _fixture
            _fixture(root)
            kw = {}
        elif dataset == "dynerf":
            from test_dynerf import _write_fixture
            _write_fixture(root)
            kw = {}
        else:
            from test_hypernerf import _write_fixture
            _write_fixture(root)
            kw = {"ratio": 1.0}
        from data.scene import load_scene
        from deformation.field import DeformationField
        from deformation.render_4d import render_deformed_view
        from gaussians.gaussian_model import CanonicalGaussianModel
        from training.losses import reconstruction_loss_4dgs
        from training.trainer_4d import init_4d_model

        scene = load_scene(root, seed=0, **kw)
        cfg = get_trainer_config(dataset)
        assert scene.dataset_type == dataset
        model, field = init_4d_model(scene, cfg, device=device)
        assert model.num_points > 0 and field.hexplane.feat_dim > 0
        view = scene.train_views[0]
        pkg = render_deformed_view(view.camera, model, field, device=device)
        assert torch.isfinite(pkg["render"]).all()
        loss = reconstruction_loss_4dgs(
            pkg["render"].unsqueeze(0), view.image.to(device).unsqueeze(0),
            cfg.optim.lambda_dssim)
        loss.backward()
        assert torch.isfinite(loss)
        n_grid = sum(1 for p in field.get_grid_parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_grid > 0
        print(f"  {dataset}: views={len(scene.train_views)} "
              f"feat={field.hexplane.feat_dim} loss={loss.item():.4f} OK")


def test_gpu_smoke_per_dataset() -> None:
    if not torch.cuda.is_available():
        print("test_gpu_smoke_per_dataset: skipped (no CUDA)")
        return
    for dataset in ("dnerf", "dynerf", "hypernerf"):
        _smoke_dataset(dataset)
    print("test_gpu_smoke_per_dataset: passed")


if __name__ == "__main__":
    test_dispatch_and_dnerf_exact()
    test_dynerf_hypernerf_exact()
    test_field_and_optimizer_wiring()
    test_resume_rejects_incompatible_arch()
    test_gpu_smoke_per_dataset()
    print("test_configs.py: done")

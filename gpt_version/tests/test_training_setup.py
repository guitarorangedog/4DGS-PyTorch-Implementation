"""CPU tests for training/losses.py, schedules.py, optim.py. Run via `python3 -m tests.test_training_setup`."""

import math

import torch

from gaussians.densify import OPTIM_GROUP_NAMES, densify_and_clone, prune_points
from gaussians.gaussian_model import CanonicalGaussianModel
from training.losses import l1_loss, reconstruction_loss, ssim
from training.optim import (
    StaticTrainingConfig,
    get_group,
    setup_static_optimizer,
    update_xyz_lr,
)
from training.schedules import get_expon_lr_func


def _model(n: int = 12, seed: int = 0) -> CanonicalGaussianModel:
    torch.manual_seed(seed)
    model = CanonicalGaussianModel(max_sh_degree=1)
    model.create_from_pointcloud(torch.randn(n, 3), torch.rand(n, 3))
    return model


def test_l1_known() -> None:
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    gt = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    assert abs(l1_loss(pred, gt).item() - 1.5) < 1e-6


def test_ssim_identity_and_degraded() -> None:
    torch.manual_seed(0)
    img = torch.rand(1, 3, 32, 32)
    assert abs(ssim(img, img).item() - 1.0) < 1e-4
    noisy = (img + 0.5 * torch.rand_like(img)).clamp(0, 1)
    s = ssim(img, noisy).item()
    assert s < 0.999, s


def test_combined_loss_formula() -> None:
    torch.manual_seed(1)
    pred = torch.rand(1, 3, 24, 24)
    gt = torch.rand(1, 3, 24, 24)
    lam = 0.2
    expected = (1 - lam) * l1_loss(pred, gt) + lam * (1 - ssim(pred, gt))
    assert torch.allclose(reconstruction_loss(pred, gt, lam), expected, atol=1e-6)
    assert torch.allclose(
        reconstruction_loss(pred, gt, 0.0), l1_loss(pred, gt), atol=1e-7
    )


def test_schedule_endpoints_and_midpoint() -> None:
    f = get_expon_lr_func(0.00016, 0.0000016, max_steps=20_000)
    assert abs(f(0) - 0.00016) < 1e-12
    assert abs(f(20_000) - 0.0000016) < 1e-12
    assert abs(f(10_000) - math.sqrt(0.00016 * 0.0000016)) < 1e-12
    assert f(30_000) == f(20_000)  # clamped past max_steps
    g = get_expon_lr_func(0.1, 0.001, lr_delay_steps=100, lr_delay_mult=0.01,
                          max_steps=1000)
    assert abs(g(0) - 0.1 * 0.01) < 1e-12
    assert abs(g(1000) - 0.001) < 1e-9


def test_optimizer_groups_and_lrs() -> None:
    model = _model()
    cfg = StaticTrainingConfig()
    opt = setup_static_optimizer(model, cfg, spatial_lr_scale=2.0)
    assert [g["name"] for g in opt.param_groups] == list(OPTIM_GROUP_NAMES)
    params = dict(model.static_param_groups())
    for g in opt.param_groups:
        assert g["params"][0] is params[g["name"]]
    assert abs(get_group(model, "xyz")["lr"] - 0.00016 * 2.0) < 1e-12
    assert abs(get_group(model, "f_dc")["lr"] - 0.0025) < 1e-12
    assert abs(get_group(model, "f_rest")["lr"] - 0.0025 / 20.0) < 1e-12
    assert abs(get_group(model, "opacity")["lr"] - 0.05) < 1e-12
    assert abs(get_group(model, "scaling")["lr"] - 0.005) < 1e-12
    assert abs(get_group(model, "rotation")["lr"] - 0.001) < 1e-12
    assert model.spatial_lr_scale == 2.0 and model.percent_dense == 0.01
    assert model.xyz_gradient_accum.shape == (12, 1)


def test_scheduler_updates_only_xyz() -> None:
    model = _model()
    setup_static_optimizer(model, spatial_lr_scale=1.0)
    before = {g["name"]: g["lr"] for g in model.optimizer.param_groups}
    lr = update_xyz_lr(model, 10_000)
    assert abs(lr - math.sqrt(0.00016 * 0.0000016)) < 1e-12
    for g in model.optimizer.param_groups:
        expected = lr if g["name"] == "xyz" else before[g["name"]]
        assert abs(g["lr"] - expected) < 1e-15, g["name"]


def test_step_then_surgery_stays_aligned() -> None:
    model = _model(n=10)
    setup_static_optimizer(model)
    loss = model._xyz.pow(2).sum() + model._opacity.pow(2).sum()
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        model._scaling.copy_(torch.full((10, 3), -6.0))
    n = densify_and_clone(model, torch.full((10, 1), 1.0), 0.5, 10.0, 0.01)
    assert n == 10 and model.num_points == 20
    model.optimizer.step()  # must not fail on post-surgery buffers
    prune_points(model, torch.tensor([True] * 5 + [False] * 15))
    assert model.num_points == 15
    for g in model.optimizer.param_groups:
        assert g["params"][0].shape[0] == 15, g["name"]
        state = model.optimizer.state.get(g["params"][0], None)
        if state is not None:
            assert state["exp_avg"].shape == g["params"][0].shape
    model.optimizer.step()


if __name__ == "__main__":
    test_l1_known()
    test_ssim_identity_and_degraded()
    test_combined_loss_formula()
    test_schedule_endpoints_and_midpoint()
    test_optimizer_groups_and_lrs()
    test_scheduler_updates_only_xyz()
    test_step_then_surgery_stays_aligned()
    print("test_training_setup.py: all 7 tests passed")

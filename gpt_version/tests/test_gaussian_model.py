"""CPU tests for gaussians/gaussian_model.py. Run via `python3 -m tests.test_gaussian_model`."""

import os
import tempfile

import torch

from gaussians.gaussian_model import (
    INIT_OPACITY,
    CanonicalGaussianModel,
    inverse_sigmoid,
    nearest_sq_distances,
)
from gaussians.sh import num_sh_coeffs, rgb_to_sh


def _cloud(n: int = 20, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n, 3), torch.rand(n, 3)


def test_parameter_shapes() -> None:
    pts, col = _cloud()
    model = CanonicalGaussianModel(max_sh_degree=3)
    model.create_from_pointcloud(pts, col)
    k = num_sh_coeffs(3)
    assert model._xyz.shape == (20, 3)
    assert model._scaling.shape == (20, 3)
    assert model._rotation.shape == (20, 4)
    assert model._opacity.shape == (20, 1)
    assert model._features_dc.shape == (20, 1, 3)
    assert model._features_rest.shape == (20, k - 1, 3)
    assert model.get_xyz.shape == (20, 3)
    assert model.get_scaling.shape == (20, 3)
    assert model.get_rotation.shape == (20, 4)
    assert model.get_opacity.shape == (20, 1)
    assert model.get_features.shape == (20, k, 3)
    assert model.get_covariance().shape == (20, 3, 3)


def test_activation_inverse() -> None:
    assert torch.allclose(
        torch.sigmoid(inverse_sigmoid(torch.tensor([INIT_OPACITY]))),
        torch.tensor([INIT_OPACITY]),
        atol=1e-6,
    )
    s = torch.tensor([[0.5, 1.0, 2.0]])
    assert torch.allclose(torch.exp(torch.log(s)), s, atol=1e-6)


def test_init_invariants() -> None:
    pts, col = _cloud()
    model = CanonicalGaussianModel(max_sh_degree=3)
    model.create_from_pointcloud(pts, col)
    assert torch.allclose(model.get_xyz, pts, atol=1e-6)
    assert torch.allclose(
        model._rotation, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(20, 1), atol=1e-6
    )
    assert torch.allclose(
        model.get_opacity, INIT_OPACITY * torch.ones(20, 1), atol=1e-6
    )
    # Isotropic scales: all 3 axes equal per point, finite.
    assert torch.allclose(model._scaling[:, 0], model._scaling[:, 1], atol=1e-6)
    assert torch.allclose(model._scaling[:, 1], model._scaling[:, 2], atol=1e-6)
    assert torch.isfinite(model._scaling).all()
    # SH: DC seeded from colors, rest zeros, degree starts at 0.
    assert torch.allclose(model._features_dc[:, 0, :], rgb_to_sh(col), atol=1e-6)
    assert (model._features_rest == 0).all()
    assert model.active_sh_degree == 0
    model.oneupSHdegree()
    assert model.active_sh_degree == 1


def test_quaternion_normalization() -> None:
    pts, col = _cloud(n=8)
    model = CanonicalGaussianModel(max_sh_degree=1)
    model.create_from_pointcloud(pts, col)
    with torch.no_grad():
        model._rotation.copy_(5.0 * torch.randn(8, 4))
    assert torch.allclose(
        model.get_rotation.norm(dim=1), torch.ones(8), atol=1e-5
    )


def test_nearest_scale_known_triangle() -> None:
    pts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    d2 = nearest_sq_distances(pts)
    assert torch.allclose(d2, torch.tensor([1.0, 1.0, 4.0]), atol=1e-5)
    model = CanonicalGaussianModel(max_sh_degree=0)
    model.create_from_pointcloud(pts, torch.full((3, 3), 0.5))
    assert torch.allclose(
        model.get_scaling[:, 0], torch.tensor([1.0, 1.0, 2.0]), atol=1e-5
    )


def test_ply_roundtrip_raw_values() -> None:
    pts, col = _cloud(n=12, seed=4)
    model = CanonicalGaussianModel(max_sh_degree=2)
    model.create_from_pointcloud(pts, col)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.ply")
        model.save_ply(path)
        restored = CanonicalGaussianModel(max_sh_degree=2).load_ply(path)
    for name in ("_xyz", "_scaling", "_rotation", "_opacity",
                 "_features_dc", "_features_rest"):
        assert torch.allclose(
            getattr(restored, name).detach(), getattr(model, name).detach(), atol=1e-6
        ), name
    assert restored.active_sh_degree == restored.max_sh_degree == 2
    assert restored.get_features.shape == model.get_features.shape


if __name__ == "__main__":
    test_parameter_shapes()
    test_activation_inverse()
    test_init_invariants()
    test_quaternion_normalization()
    test_nearest_scale_known_triangle()
    test_ply_roundtrip_raw_values()
    print("test_gaussian_model.py: all 6 tests passed")

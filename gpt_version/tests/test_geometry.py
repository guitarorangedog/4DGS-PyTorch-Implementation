"""CPU tests for gaussians/geometry.py. Runnable via `python3 tests/test_geometry.py`."""

import math

import torch

from gaussians.geometry import (
    build_covariance,
    build_rotation,
    build_scaling_rotation,
    focal2fov,
    fov2focal,
    get_projection_matrix,
    normalize_quaternion,
    pack_symmetric,
    unpack_symmetric,
)


def test_identity_quaternion_gives_identity() -> None:
    R = build_rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert R.shape == (1, 3, 3)
    assert torch.allclose(R[0], torch.eye(3), atol=1e-6)


def test_rotation_orthonormal_and_det_one() -> None:
    torch.manual_seed(1)
    q = torch.randn(8, 4)
    R = build_rotation(q)
    assert R.shape == (8, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(8, 3, 3), atol=1e-5)
    assert torch.allclose(R.det(), torch.ones(8), atol=1e-5)
    # Unnormalized input must give the same rotation as normalized input.
    assert torch.allclose(R, build_rotation(3.0 * q), atol=1e-5)


def test_90deg_z_rotation() -> None:
    # +90 deg about z: (x, y) -> (-y, x). Quaternion (cos45, 0, 0, sin45).
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
    R = build_rotation(q)[0]
    v = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(R @ v, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_covariance_psd_and_pack_roundtrip() -> None:
    torch.manual_seed(2)
    N = 6
    s = torch.exp(torch.randn(N, 3))  # activated scales, strictly positive
    q = normalize_quaternion(torch.randn(N, 4))
    L = build_scaling_rotation(s, q)
    assert L.shape == (N, 3, 3)
    cov = build_covariance(s, q)
    assert cov.shape == (N, 3, 3)
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-6)
    assert (torch.linalg.eigvalsh(cov) > 0).all()
    # Isotropic case: Sigma = s^2 * I.
    cov_iso = build_covariance(
        torch.tensor([[2.0, 2.0, 2.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    assert torch.allclose(cov_iso[0], 4.0 * torch.eye(3), atol=1e-5)
    # Pack/unpack roundtrip in rasterizer 6-vector order.
    packed = pack_symmetric(cov)
    assert packed.shape == (N, 6)
    assert torch.allclose(unpack_symmetric(packed), cov, atol=1e-6)


def test_fov_focal_roundtrip() -> None:
    for fov in (0.5, 1.0, 1.5):
        assert abs(focal2fov(fov2focal(fov, 800.0), 800.0) - fov) < 1e-9


def test_projection_matrix_sane() -> None:
    P = get_projection_matrix(0.1, 100.0, fovX=1.0, fovY=0.8)
    assert P.shape == (4, 4)
    assert P[3, 2].item() == 1.0
    assert P[0, 0].item() > 0 and P[1, 1].item() > 0
    # Symmetric frustum => zero off-center terms.
    assert P[0, 2].item() == 0.0 and P[1, 2].item() == 0.0


if __name__ == "__main__":
    test_identity_quaternion_gives_identity()
    test_rotation_orthonormal_and_det_one()
    test_90deg_z_rotation()
    test_covariance_psd_and_pack_roundtrip()
    test_fov_focal_roundtrip()
    test_projection_matrix_sane()
    print("test_geometry.py: all 6 tests passed")

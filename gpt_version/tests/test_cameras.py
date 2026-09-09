"""CPU tests for data/cameras.py. Runnable via `python3 -m tests.test_cameras`."""

import math

import numpy as np
import torch

from data.cameras import Camera, get_world_to_view


def _make(FoVx: float = 1.0, FoVy: float = 0.8, W: int = 800, H: int = 600,
          R=None, T=None, time: float = 0.0) -> Camera:
    if R is None:
        R = np.eye(3)
    if T is None:
        T = np.zeros(3)
    return Camera(R=R, T=T, FoVx=FoVx, FoVy=FoVy,
                  image_width=W, image_height=H, time=time)


def test_transform_shapes() -> None:
    cam = _make()
    assert cam.world_view_transform.shape == (4, 4)
    assert cam.projection_matrix.shape == (4, 4)
    assert cam.full_proj_transform.shape == (4, 4)
    assert cam.camera_center.shape == (3,)
    assert cam.R.shape == (3, 3) and cam.T.shape == (3,)


def test_identity_camera() -> None:
    cam = _make()
    # Row-major builder gives identity, transpose of identity is identity.
    assert torch.allclose(cam.world_view_transform,
                          torch.eye(4), atol=1e-6)
    assert torch.allclose(cam.camera_center,
                          torch.zeros(3), atol=1e-6)
    # Composition rule used by the rasterizer setup.
    assert torch.allclose(cam.full_proj_transform,
                          cam.world_view_transform @ cam.projection_matrix,
                          atol=1e-6)


def test_transpose_convention() -> None:
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    T = rng.normal(size=3)
    cam = _make(R=Q, T=T)
    # world_view_transform.T is the classical [R^T | T] matrix.
    w2v_T = cam.world_view_transform.transpose(0, 1)
    assert torch.allclose(w2v_T[:3, :3],
                          torch.tensor(Q.T, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(w2v_T[:3, 3],
                          torch.tensor(T, dtype=torch.float32), atol=1e-5)
    # ...so the stored (non-transposed) block equals R itself.
    assert torch.allclose(cam.world_view_transform[:3, :3],
                          torch.tensor(Q, dtype=torch.float32), atol=1e-5)


def test_camera_center_translation() -> None:
    # Camera at C=(0,0,5), R=I  <=>  T = -R^T @ C = (0,0,-5).
    cam = _make(T=np.array([0.0, 0.0, -5.0]))
    assert torch.allclose(cam.camera_center,
                          torch.tensor([0.0, 0.0, 5.0]), atol=1e-5)
    # World origin lands at (0,0,-5) in camera space.
    assert torch.allclose(cam.world_to_camera(torch.zeros(1, 3)),
                          torch.tensor([[0.0, 0.0, -5.0]]), atol=1e-5)
    # And the camera center itself maps to the origin.
    assert torch.allclose(
        cam.world_to_camera(cam.camera_center.unsqueeze(0)),
        torch.zeros(1, 3), atol=1e-5)


def test_rotated_camera_center_stays_origin_mapped() -> None:
    # +90 deg about z stored as R; center still T=0 -> 0.
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cam = _make(R=Rz, T=np.zeros(3))
    assert torch.allclose(cam.camera_center, torch.zeros(3), atol=1e-5)
    # x-axis maps under R^T (= Rz^T): (1,0,0) -> (c,-s,0).
    out = cam.world_to_camera(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[c, -s, 0.0]]), atol=1e-5)


def test_time_passthrough() -> None:
    cam = _make(time=0.37)
    assert cam.time == 0.37


def test_helper_matches_classical_form() -> None:
    R = np.eye(3)
    T = np.array([1.0, 2.0, 3.0])
    Rt = get_world_to_view(R, T)
    assert Rt.shape == (4, 4)
    assert np.allclose(Rt[:3, :3], R.T, atol=1e-6)
    assert np.allclose(Rt[:3, 3], T, atol=1e-6)


if __name__ == "__main__":
    test_transform_shapes()
    test_identity_camera()
    test_transpose_convention()
    test_camera_center_translation()
    test_rotated_camera_center_stays_origin_mapped()
    test_time_passthrough()
    test_helper_matches_classical_form()
    print("test_cameras.py: all 7 tests passed")

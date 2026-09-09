"""Camera representation for 3D-GS / 4D-GS [STATIC infra-free].

Exposes exactly what the renderer and (later) dataset readers need::

    R, T, FoVx, FoVy, image_width, image_height, time,
    world_view_transform, projection_matrix, full_proj_transform, camera_center.

Concept reference: paper Sec. 3.1 (splatting projects Gaussians into the
camera plane); the render call signature is ``G(S' | R, T)`` (paper Eq. 8).

Coordinate / matrix convention (matches official 3D-GS / 4DGaussians
``scene/cameras.py`` + ``utils/graphics_utils.py`` exactly):

- ``R`` (``[3, 3]``) and ``T`` (``[3]``) are the COLMAP-style stored pair.
  The world-to-camera rotation is ``R^T`` — that is why the builder below
  writes ``Rt[:3, :3] = R^T`` (see ``getWorld2View2``). ``T`` is the
  translation part of the 4x4 world-to-view matrix, i.e. a world point maps
  as ``x_cam = R^T @ x_world + T`` (equivalently ``T = -R^T @ C`` for camera
  center ``C``).
- Matrices are built row-major in NumPy (``Rt``), then stored transposed
  (``world_view_transform = Rt^T``) because the CUDA rasterizer consumes
  the transposed layout. Concretely: ``world_view_transform.T`` equals the
  classical ``[R^T | T; 0 1]`` matrix, and its ``[:3, :3]`` block equals the
  stored ``R`` only after accounting for this transpose
  (``W2V[:3,:3] == R``, ``W2V.T[:3,:3] == R^T``).
- ``projection_matrix`` is ``getProjectionMatrix(znear, zfar, FoVx, FoVy)^T``
  (same transpose rule; reuses ``gaussians.geometry.get_projection_matrix``).
- ``full_proj_transform = world_view_transform @ projection_matrix``.
- ``camera_center = inverse(world_view_transform)[3, :3]``.
- Multiplication convention for points: row vectors, ``x_out = x_hom @ W2V``
  with ``x_hom = [x, 1]``. Equivalently the ``3x3`` part acts as
  ``x_cam = R^T @ x + T``.
- ``time`` is a normalized timestamp in ``[0, 1]`` (frame index /
  ``num_frames``). It is stored here so the 4D deformation field can later
  query ``(X, t)`` per Gaussian; no deformation behavior lives in this
  module.

This module is deliberately independent of dataset readers,
``GaussianModel``, the rasterizer, deformation, and training. All tensors
are CPU ``float32`` here; callers move them to CUDA at render time
(``.cuda()``), mirroring the official renderer.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from gaussians.geometry import get_projection_matrix

__all__ = ["Camera", "get_world_to_view"]


def get_world_to_view(
    R: np.ndarray,
    T: np.ndarray,
    translate: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale: float = 1.0,
) -> np.ndarray:
    """Build the 4x4 world-to-view matrix, row-major (pre-transpose).

    Exact mirror of official ``utils/graphics_utils.getWorld2View2``.

    Args:
        R: ``[3, 3]`` stored rotation (world-to-camera rotation is ``R^T``).
        T: ``[3]`` translation part.
        translate: scene-normalization offset applied to the camera center.
        scale: scene-normalization scale applied to the camera center.

    Returns:
        ``[4, 4]`` float32 ``Rt`` with ``Rt[:3, :3] = R^T``, ``Rt[:3, 3] = T``
        (before the rasterizer transpose), adjusted by ``translate``/``scale``
        through the camera center.
    """
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = np.asarray(R).transpose()
    Rt[:3, 3] = np.asarray(T, dtype=np.float64)
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + np.asarray(translate)) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


@dataclass
class Camera:
    """Single pinhole view + timestamp.

    Attributes:
        R: ``[3, 3]`` stored rotation (see module docstring for transpose rule).
        T: ``[3]`` translation part of the world-to-view matrix.
        FoVx: horizontal field of view, radians.
        FoVy: vertical field of view, radians.
        image_width: image width in pixels (``W``).
        image_height: image height in pixels (``H``).
        time: normalized timestamp in ``[0, 1]``; deformation input later.
        znear: near plane (default ``0.01``, official value).
        zfar: far plane (default ``100.0``, official value).
        world_view_transform: ``[4, 4]`` transposed W2V (rasterizer layout).
        projection_matrix: ``[4, 4]`` transposed projection.
        full_proj_transform: ``[4, 4]`` = ``world_view_transform @ projection``.
        camera_center: ``[3]`` world-space camera center.
    """

    R: torch.Tensor
    T: torch.Tensor
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    time: float = 0.0
    znear: float = 0.01
    zfar: float = 100.0
    trans: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    scale: float = 1.0
    world_view_transform: torch.Tensor = field(init=False)
    projection_matrix: torch.Tensor = field(init=False)
    full_proj_transform: torch.Tensor = field(init=False)
    camera_center: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        R = torch.as_tensor(self.R, dtype=torch.float32)
        T = torch.as_tensor(self.T, dtype=torch.float32)
        assert R.shape == (3, 3), f"R must be [3, 3], got {tuple(R.shape)}"
        assert T.shape == (3,), f"T must be [3], got {tuple(T.shape)}"
        assert self.image_width > 0 and self.image_height > 0
        self.R = R
        self.T = T
        self.time = float(self.time)

        w2v = torch.tensor(
            get_world_to_view(
                self.R.numpy(), self.T.numpy(),
                np.asarray(self.trans, dtype=np.float64), self.scale,
            ),
            dtype=torch.float32,
        ).transpose(0, 1)
        proj = get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1)

        self.world_view_transform = w2v  # [4, 4]
        self.projection_matrix = proj  # [4, 4]
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix  # [4, 4]
        self.camera_center = self.world_view_transform.inverse()[3, :3]  # [3]

    def world_to_camera(self, points: torch.Tensor) -> torch.Tensor:
        """Map world points to camera space (row-vector convention).

        Args:
            points: ``[N, 3]`` world coordinates.

        Returns:
            ``[N, 3]`` camera coordinates: ``R^T @ x + T``.
        """
        assert points.shape[-1] == 3, f"expected [N, 3], got {tuple(points.shape)}"
        R_w2c = self.R.transpose(0, 1)
        return points @ R_w2c.transpose(0, 1) + self.T

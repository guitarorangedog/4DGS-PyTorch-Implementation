"""Gaussian geometry / covariance / projection math for 3D-GS [STATIC].

Covers paper Sec. 3.1, Eqs. 1-3:

- Eq. 1: ``G(X) = exp(-1/2 X^T Sigma^-1 X)`` — anisotropic Gaussian.
- Eq. 2: ``Sigma = R S S^T R^T`` — covariance from rotation ``R`` and
  scale ``S``. We store per-Gaussian scale ``s in R^3`` (log-space in the
  model) and unit quaternion ``r in R^4`` (w, x, y, z order).
- Eq. 3: ``Sigma' = J W Sigma W^T J^T`` — EWA projection to camera space.
  The 2D projection itself (Jacobian ``J``, blending) lives in the CUDA
  rasterizer (Commit 6); this module provides the 3D covariance side plus
  the camera-projection helpers (focal/fov, projection matrix).

Conventions match the official 3D-GS / 4DGaussians code
(``utils/general_utils.build_rotation / build_scaling_rotation /
strip_symmetric`` and ``utils/graphics_utils.fov2focal / focal2fov /
getProjectionMatrix``) with one documented deviation: the official code
hardcodes ``device="cuda"`` tensors; every function here is device-agnostic
(``x.device``) so CPU-only math tests can run. The formulas are unchanged.

All functions are small, pure PyTorch, batched over ``N`` Gaussians.
"""

import math

import torch

__all__ = [
    "normalize_quaternion",
    "build_rotation",
    "build_scaling_rotation",
    "build_covariance",
    "pack_symmetric",
    "unpack_symmetric",
    "fov2focal",
    "focal2fov",
    "get_projection_matrix",
]


def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit length.

    Args:
        q: ``[N, 4]`` quaternions in ``(w, x, y, z)`` order.

    Returns:
        ``[N, 4]`` unit quaternions.
    """
    return q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_rotation(q: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions (paper Eq. 2, ``R``).

    Formula identical to ``general_utils.build_rotation``.

    Args:
        q: ``[N, 4]`` quaternions ``(w, x, y, z)`` (need not be normalized;
            normalized internally).

    Returns:
        ``[N, 3, 3]`` rotation matrices.
    """
    assert q.shape[-1] == 4, f"expected [N, 4], got {tuple(q.shape)}"
    q = normalize_quaternion(q)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Build the ``L = R @ diag(s)`` factor (paper Eq. 2).

    ``Sigma = L @ L^T``. Formula identical to
    ``general_utils.build_scaling_rotation``.

    Args:
        s: ``[N, 3]`` scales (already activated, i.e. world units —
            the model stores log-scales and applies ``exp`` elsewhere).
        q: ``[N, 4]`` quaternions ``(w, x, y, z)``.

    Returns:
        ``[N, 3, 3]`` matrix ``L``.
    """
    assert s.shape[-1] == 3, f"expected [N, 3], got {tuple(s.shape)}"
    assert q.shape == (s.shape[0], 4), f"mismatched {tuple(s.shape)} vs {tuple(q.shape)}"
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(q)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def build_covariance(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Build full 3D covariance matrices (paper Eq. 2).

    Args:
        s: ``[N, 3]`` activated scales.
        q: ``[N, 4]`` quaternions ``(w, x, y, z)``.

    Returns:
        ``[N, 3, 3]`` symmetric positive semi-definite covariances
        ``Sigma = L @ L^T`` with ``L = R @ diag(s)``.
    """
    L = build_scaling_rotation(s, q)
    return L @ L.transpose(1, 2)


def pack_symmetric(mat: torch.Tensor) -> torch.Tensor:
    """Pack symmetric 3x3 matrices into 6-vectors (rasterizer convention).

    Element order matches ``general_utils.strip_symmetric`` / the CUDA
    rasterizer's ``cov3D`` layout: ``[m00, m01, m02, m11, m12, m22]``.

    Args:
        mat: ``[N, 3, 3]`` symmetric matrices.

    Returns:
        ``[N, 6]`` packed vectors.
    """
    assert mat.shape[-2:] == (3, 3), f"expected [N, 3, 3], got {tuple(mat.shape)}"
    out = torch.zeros((mat.shape[0], 6), dtype=mat.dtype, device=mat.device)
    out[:, 0] = mat[:, 0, 0]
    out[:, 1] = mat[:, 0, 1]
    out[:, 2] = mat[:, 0, 2]
    out[:, 3] = mat[:, 1, 1]
    out[:, 4] = mat[:, 1, 2]
    out[:, 5] = mat[:, 2, 2]
    return out


def unpack_symmetric(packed: torch.Tensor) -> torch.Tensor:
    """Unpack 6-vectors back to symmetric 3x3 matrices.

    Inverse of :func:`pack_symmetric`.

    Args:
        packed: ``[N, 6]`` in ``[m00, m01, m02, m11, m12, m22]`` order.

    Returns:
        ``[N, 3, 3]`` symmetric matrices.
    """
    assert packed.shape[-1] == 6, f"expected [N, 6], got {tuple(packed.shape)}"
    mat = torch.zeros((packed.shape[0], 3, 3), dtype=packed.dtype, device=packed.device)
    mat[:, 0, 0] = packed[:, 0]
    mat[:, 0, 1] = packed[:, 1]
    mat[:, 1, 0] = packed[:, 1]
    mat[:, 0, 2] = packed[:, 2]
    mat[:, 2, 0] = packed[:, 2]
    mat[:, 1, 1] = packed[:, 3]
    mat[:, 1, 2] = packed[:, 4]
    mat[:, 2, 1] = packed[:, 4]
    mat[:, 2, 2] = packed[:, 5]
    return mat


def fov2focal(fov: float, pixels: float) -> float:
    """Field-of-view (radians) to focal length in pixels.

    Same as ``graphics_utils.fov2focal``: ``pixels / (2 tan(fov / 2))``.
    """
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: float) -> float:
    """Focal length in pixels to field-of-view in radians.

    Same as ``graphics_utils.focal2fov``: ``2 atan(pixels / (2 focal))``.
    """
    return 2 * math.atan(pixels / (2 * focal))


def get_projection_matrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    """Build the OpenGL-style projection matrix used by 3D-GS.

    Formula identical to ``graphics_utils.getProjectionMatrix``.
    Concept: maps camera-space points into the clip space consumed by the
    splatting projection (paper Eq. 3 setup; the Jacobian ``J`` and
    ``Sigma' = J W Sigma W^T J^T`` are applied inside the rasterizer).

    Args:
        znear: near plane distance.
        zfar: far plane distance.
        fovX: horizontal field of view in radians.
        fovY: vertical field of view in radians.

    Returns:
        ``[4, 4]`` float32 projection matrix (CPU tensor).
    """
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, dtype=torch.float32)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

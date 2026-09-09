"""Spherical Harmonics (SH) utilities for 3D-GS [STATIC].

Covers the color part of the 3D-GS formulation (paper Sec. 3.1, Eq. 4):

- Each Gaussian stores view-dependent color as SH coefficients
  ``C in R^k`` with ``k = 3 * (deg + 1) ** 2`` (one band set per RGB channel).
- The target implementation uses ``max_sh_degree = 3`` (see official
  ``arguments/__init__.py`` ``ModelParams.sh_degree``), so only degrees 0-3
  are supported here. Degree 4 present in the upstream PlenOctree port is
  intentionally omitted (unused, keeps the port minimal).

Conventions match ``utils/sh_utils.py`` of 3D-GS / 4DGaussians exactly
(same polynomial constants ``C0..C3`` and DC conversions).

All functions are pure PyTorch and device-agnostic (CPU and CUDA).
"""

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]

MAX_SH_DEGREE = 3


def num_sh_coeffs(degree: int) -> int:
    """Number of SH coefficients per channel for a given degree.

    Args:
        degree: SH degree in ``[0, 3]``.

    Returns:
        ``(degree + 1) ** 2``, e.g. 1 / 4 / 9 / 16.
    """
    assert 0 <= degree <= MAX_SH_DEGREE, f"degree {degree} not in [0, 3]"
    return (degree + 1) ** 2


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB color to SH DC component.

    Inverse of :func:`sh_to_rgb`. Used at point-cloud initialization
    (``create_from_pcd`` in the official code) to seed ``features_dc``.

    Args:
        rgb: ``[..., 3]`` values in ``[0, 1]``.

    Returns:
        ``[..., 3]`` SH DC coefficients: ``(rgb - 0.5) / C0``.
    """
    return (rgb - 0.5) / C0


def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert SH DC component back to linear RGB.

    Args:
        sh_dc: ``[..., 3]`` DC coefficients.

    Returns:
        ``[..., 3]`` colors: ``sh_dc * C0 + 0.5``.
    """
    return sh_dc * C0 + 0.5


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH color for unit view directions.

    Hardcoded SH polynomials, identical to the upstream 3D-GS port
    (PlenOctree convention). Concept reference: paper Eq. 4, where each
    Gaussian's color ``c_i`` comes from its SH coefficients.

    Args:
        degree: active SH degree in ``[0, 3]``.
        sh: ``[..., 3, (degree + 1) ** 2]`` SH coefficients.
        dirs: ``[..., 3]`` unit view directions (gaussian -> camera,
            normalized). Must be broadcastable against ``sh``.

    Returns:
        ``[..., 3]`` view-dependent colors (linear, before ``+0.5`` shift
        used by some rasterizer paths; see note below).

    Note:
        Degree 0 reduces to ``C0 * sh[..., 0]``. The CUDA rasterizer adds
        the ``+0.5`` base internally (``sh2rgb + 0.5``); this function
        returns the raw polynomial sum, matching ``utils/sh_utils.eval_sh``.
    """
    assert 0 <= degree <= MAX_SH_DEGREE, f"degree {degree} not in [0, 3]"
    assert sh.shape[-2] == 3, f"expected [..., 3, K], got {tuple(sh.shape)}"
    assert sh.shape[-1] >= (degree + 1) ** 2
    assert dirs.shape[-1] == 3, f"expected [..., 3], got {tuple(dirs.shape)}"

    result = C0 * sh[..., 0]
    if degree > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result
            - C1 * y * sh[..., 1]
            + C1 * z * sh[..., 2]
            - C1 * x * sh[..., 3]
        )

        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if degree > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
    return result

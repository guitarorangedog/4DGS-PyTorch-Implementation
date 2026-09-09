"""Faithful CUDA Gaussian rasterizer binding [STATIC].

Thin wrapper around the official CUDA extension
(``depth-diff-gaussian-rasterization`` family, see ``setup_rasterizer.md``).
It connects :class:`CanonicalGaussianModel` + :class:`Camera` to
``GaussianRasterizer`` using the exact settings conventions of the official
``gaussian_renderer/__init__.py`` static path (no deformation yet):

- ``screenspace_points = zeros_like(xyz, requires_grad=True)`` so gradients
  flow back to 2D means (used by densification statistics in Commit 5).
- ``tanfov`` from ``FoVx/FoVy``; ``viewmatrix/projmatrix`` moved to CUDA.
- Rasterizer inputs: ``means3D`` activated positions, **raw** log-scales and
  **raw** quaternions (the CUDA kernels apply ``exp``/normalize internally),
  **activated** opacities (``sigmoid``), **raw** SH coefficients.
- Outputs: ``render [3, H, W]``, ``viewspace_points [N, 3]``,
  ``visibility_filter [N]`` (``radii > 0``), ``radii [N]``, ``depth [H, W]``.

Hard rules (Commit 6 gate):

- No pure-PyTorch / simplified rendering fallback anywhere in this repo.
  If the extension (or CUDA) is unavailable, :func:`render_view` raises
  ``RuntimeError`` immediately. Math modules stay importable regardless.
"""

import math

import torch

__all__ = ["RASTERIZER_SPEC", "require_rasterizer", "render_view"]

#: Pinned faithful source. Installed separately (NOT via requirements.txt).
RASTERIZER_SPEC = (
    "ingra14m/depth-diff-gaussian-rasterization @ 9055fcf (2023-11-09), "
    "the exact submodule pinned by hustvl/4DGaussians, plus one build-only "
    "patch: `#include <cstdint>` in cuda_rasterizer/rasterizer_impl.h "
    "(newer nvcc no longer provides uint32_t transitively; no math change)."
)


def require_rasterizer():
    """Import the CUDA extension or raise a clear ``RuntimeError``.

    Returns:
        The ``diff_gaussian_rasterization`` module.

    Raises:
        RuntimeError: if CUDA is unavailable or the extension is not built.
            See ``setup_rasterizer.md`` for build instructions.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "render_view requires CUDA, but torch.cuda.is_available() is False."
        )
    try:
        import diff_gaussian_rasterization as ext
    except ImportError as e:
        raise RuntimeError(
            "CUDA rasterizer extension 'diff_gaussian_rasterization' is not "
            "installed (see setup_rasterizer.md). Refusing to fall back to "
            "any simplified renderer."
        ) from e
    return ext


def render_view(camera, model, bg_color=(1.0, 1.0, 1.0), scaling_modifier: float = 1.0,
                device: torch.device | str = "cuda") -> dict:
    """Splatter the canonical Gaussians into ``camera`` (static 3D-GS path).

    Args:
        camera: :class:`Camera` (see ``data/cameras.py`` for conventions).
        model: :class:`CanonicalGaussianModel` with parameters on ``device``.
        bg_color: ``[3]`` background color (tuple or tensor).
        scaling_modifier: global scale multiplier (official ``scaling_modifier``).
        device: CUDA device string.

    Returns:
        Dict with ``render [3, H, W]``, ``viewspace_points [N, 3]``
        (``.grad`` populated after ``backward``), ``visibility_filter [N]``
        bool, ``radii [N]`` and ``depth [H, W]`` (``None`` if the linked
        rasterizer build does not return depth).
    """
    ext = require_rasterizer()
    device = torch.device(device)

    screenspace_points = torch.zeros_like(
        model.get_xyz, dtype=model.get_xyz.dtype, requires_grad=True, device=device
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    if isinstance(bg_color, torch.Tensor):
        bg = bg_color.to(dtype=torch.float32, device=device)
    else:
        bg = torch.tensor(bg_color, dtype=torch.float32, device=device)

    raster_settings = ext.GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform.to(device),
        projmatrix=camera.full_proj_transform.to(device),
        sh_degree=int(model.active_sh_degree),
        campos=camera.camera_center.to(device),
        prefiltered=False,
        debug=False,
    )
    rasterizer = ext.GaussianRasterizer(raster_settings=raster_settings)

    means3D = model.get_xyz
    opacity = model.get_opacity
    shs = model.get_features

    out = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=model._scaling,
        rotations=model._rotation,
        cov3D_precomp=None,
    )
    if len(out) == 3:
        rendered_image, radii, depth = out
    else:  # non-depth faithful forks return (image, radii)
        rendered_image, radii = out
        depth = None

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
    }

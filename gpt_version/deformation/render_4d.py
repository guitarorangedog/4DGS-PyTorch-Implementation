"""Time-conditioned rendering: the STATIC -> 4D seam (Commit 14) [4D].

STATIC (``gaussians/rasterizer.render_view``)::

    canonical --[exp/normalize/sigmoid]--> SAME CUDA rasterizer

4D (here)::

    canonical + camera.time --[DeformationField]--> deformed (raw)
        --[exp/normalize/sigmoid]--> SAME CUDA rasterizer

The seam is exactly: broadcast ``camera.time`` -> ``field`` -> activate ->
:func:`rasterize_final`. Canonical ``nn.Parameter``s are never mutated.
Opacity/SH stay canonical (official default ``no_do``/``no_dshs``).
"""

import torch

from deformation.field import DeformationField, DeformedState
from gaussians.geometry import normalize_quaternion
from gaussians.rasterizer import rasterize_final

__all__ = ["render_deformed_view"]


def render_deformed_view(camera, model, field: DeformationField,
                         bg_color=(1.0, 1.0, 1.0), scaling_modifier: float = 1.0,
                         device: torch.device | str = "cuda") -> dict:
    """Render canonical Gaussians deformed at ``camera.time`` (4D path).

    Mirrors the official renderer ``fine`` stage: raw
    ``(get_xyz, _scaling, _rotation, _opacity, get_features)`` enter the
    field with ``time = full([N, 1], camera.time)``; the raw deformed state
    is activated (``exp``/``normalize``/``sigmoid``) before the shared CUDA
    call — the same activation point as the static path.

    Returns:
        :func:`rasterize_final` dict plus ``"state"`` (:class:`DeformedState`,
        raw representations) for debugging/testing.
    """
    device = torch.device(device)
    n = model.num_points
    time = torch.full((n, 1), float(camera.time),
                      dtype=torch.float32, device=device)
    state: DeformedState = field(
        model.get_xyz, model._scaling, model._rotation, time,
        opacity=model._opacity, shs=model.get_features,
    )
    pkg = rasterize_final(
        camera,
        means3D=state.xyz,
        scales=torch.exp(state.scaling),
        rotations=normalize_quaternion(state.rotation),
        opacity=torch.sigmoid(state.opacity),
        shs=state.shs,
        sh_degree=int(model.active_sh_degree),
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        device=device,
    )
    pkg["state"] = state
    return pkg

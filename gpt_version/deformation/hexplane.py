"""Multi-resolution HexPlane spatio-temporal encoder [4D].

Paper Sec. 4.2, Eq. 10: nearby Gaussians share similar motion, so each
Gaussian's ``(X, t)`` queries a decomposed 4D voxel grid built from six 2D
planes. This module ends at feature extraction (``f_voxel``); the Commit 12
decoder maps features to ``(dX, ds, dr)``.

Faithful to official ``scene/hexplane.py`` (``HexPlaneField``,
``init_grid_param``, ``interpolate_ms_features``, ``grid_sample_wrapper``,
``normalize_aabb``). All functions are device-agnostic (official code is
identical math, CUDA-tensor-placed).

Plane ordering (fixed by ``itertools.combinations(range(4), 2)``)::

    0: (x, y)   spatial
    1: (x, z)   spatial
    2: (x, t)   space-time
    3: (y, z)   spatial
    4: (y, t)   space-time
    5: (z, t)   space-time

Fusion rule (exact official behavior): per resolution level, bilinearly
sample all six planes and take the elementwise PRODUCT; CONCATENATE the
per-level products across resolutions. Output dim ``F = out_dim * #levels``.
"""

import itertools
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "HexPlaneConfig",
    "PLANE_NAMES",
    "PLANE_PAIRS",
    "normalize_aabb",
    "grid_sample_wrapper",
    "init_grid_param",
    "interpolate_ms_features",
    "HexPlaneField",
]

#: Stable plane order; index i <=> PLANE_PAIRS[i] axis pair.
PLANE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
PLANE_NAMES = ("xy", "xz", "xt", "yz", "yt", "zt")
#: Indices of the three space-time planes (pairs containing axis 3).
TIME_PLANE_IDS = (2, 4, 5)


@dataclass
class HexPlaneConfig:
    """Official ``ModelHiddenParams`` HexPlane defaults."""

    bounds: float = 1.6
    grid_dimensions: int = 2
    input_coordinate_dim: int = 4
    output_coordinate_dim: int = 32
    resolution: list = field(default_factory=lambda: [64, 64, 64, 25])
    multires: list = field(default_factory=lambda: [1, 2, 4, 8])


def normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    """Map spatial points into the ``[-1, 1]`` grid-sample domain.

    ``(pts - aabb[0]) * (2 / (aabb[1] - aabb[0])) - 1`` (official formula).
    Note the official ``set_aabb(xyz_max, xyz_min)`` stores
    ``aabb = [max, min]``, so the max corner maps to ``-1`` and the min
    corner to ``+1`` — inverted but self-consistent since every query uses
    the same mapping. Time is NOT normalized (stays in ``[0, 1]``, sampling
    the upper half of each time axis with ``border`` padding outside).
    """
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor,
                        align_corners: bool = True) -> torch.Tensor:
    """Bilinear 2D sampling with border padding (official wrapper).

    Args:
        grid: ``[F, Hy, Hx]`` or ``[1, F, Hy, Hx]`` plane features.
        coords: ``[N, 2]`` sample locations in ``[-1, 1]`` (x first).

    Returns:
        ``[N, F]`` sampled features (official squeezes the batch dim).
    """
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim != 2:
        raise NotImplementedError(f"only 2D planes supported, got {grid_dim}D")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(grid, coords, align_corners=align_corners,
                           mode="bilinear", padding_mode="border")
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)
    return interp.squeeze()


def init_grid_param(grid_nd: int, in_dim: int, out_dim: int,
                    reso: list, a: float = 0.1, b: float = 0.5) -> nn.ParameterList:
    """Create one level's planes (official ``init_grid_param``).

    Plane ``i`` covers axes ``PLANE_PAIRS[i]`` with shape
    ``[1, out_dim, reso[j], reso[i]]`` (note reversed axis order: the grid's
    last dim maps the pair's FIRST axis, matching ``grid_sample``'s
    ``(x, y)`` convention). Planes containing the time axis are initialized
    to ONES (identity-ish start for temporal modulation); pure spatial
    planes use ``Uniform(0.1, 0.5)``.
    """
    assert in_dim == len(reso), "resolution needs one entry per input dim"
    assert grid_nd <= in_dim
    has_time_planes = in_dim == 4
    coefs = nn.ParameterList()
    for coo_comb in itertools.combinations(range(in_dim), grid_nd):
        param = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(param)
        else:
            nn.init.uniform_(param, a=a, b=b)
        coefs.append(param)
    return coefs


def interpolate_ms_features(pts: torch.Tensor, ms_grids, concat_features: bool = True,
                            num_levels: Optional[int] = None) -> torch.Tensor:
    """Product-over-planes per level, concat over levels (official rule).

    Args:
        pts: ``[N, 4]`` normalized ``(x, y, z, t)`` coordinates.
        ms_grids: list (per level) of 6-plane ``ParameterList``s.

    Returns:
        ``[N, out_dim * #levels]`` fused features.
    """
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
    if num_levels is None:
        num_levels = len(ms_grids)
    per_level = []
    for grid in ms_grids[:num_levels]:
        acc = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feat_dim = grid[ci].shape[1]
            plane_feat = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feat_dim)
            acc = acc * plane_feat
        per_level.append(acc)
    if concat_features:
        return torch.cat(per_level, dim=-1)
    out = per_level[0]
    for extra in per_level[1:]:
        out = out + extra
    return out


class HexPlaneField(nn.Module):
    """Multi-resolution HexPlane encoder: ``(xyz, time) -> f_voxel``.

    Attribute names (``aabb``, ``grids``) intentionally match the official
    class so ``state_dict``s are interchangeable for parity checks.
    """

    def __init__(self, config: HexPlaneConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = HexPlaneConfig()
        self.config = config
        # Official initial AABB is [max-ish, min-ish] = [+b, -b]; replaced by
        # set_aabb(xyz_max, xyz_min) from the Scene before training.
        aabb = torch.tensor([[config.bounds] * 3, [-config.bounds] * 3])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in config.multires:
            # Multi-resolution applies to SPATIAL axes only; the time axis
            # keeps its base resolution (official "resolution fix").
            reso = [r * res for r in config.resolution[:3]] + config.resolution[3:]
            level = init_grid_param(
                grid_nd=config.grid_dimensions,
                in_dim=config.input_coordinate_dim,
                out_dim=config.output_coordinate_dim,
                reso=reso,
            )
            self.feat_dim += level[-1].shape[1]
            self.grids.append(level)

    @property
    def get_aabb(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(aabb[0], aabb[1])`` (official ordering: max, then min)."""
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        """Install the scene point-cloud bounds (official arg order)."""
        aabb = torch.stack([
            torch.as_tensor(xyz_max, dtype=torch.float32),
            torch.as_tensor(xyz_min, dtype=torch.float32),
        ])
        self.aabb = nn.Parameter(aabb, requires_grad=False)

    def get_grid_parameters(self) -> list[nn.Parameter]:
        """All plane parameters (for the grid optimizer group + TV losses)."""
        return [p for level in self.grids for p in level]

    def forward(self, xyz: torch.Tensor, time) -> torch.Tensor:
        """Encode points at timestamps.

        Args:
            xyz: ``[N, 3]`` spatial coordinates (world units).
            time: ``[N, 1]`` / ``[N]`` normalized timestamps in ``[0, 1]``,
                or a scalar broadcast to all points.

        Returns:
            ``[N, feat_dim]`` voxel features.
        """
        assert xyz.dim() == 2 and xyz.shape[-1] == 3, f"expected [N, 3], got {tuple(xyz.shape)}"
        t = torch.as_tensor(time, dtype=xyz.dtype, device=xyz.device)
        if t.dim() == 0:
            t = t.expand(xyz.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)
        assert t.shape == (xyz.shape[0], 1), f"expected [{xyz.shape[0]}, 1], got {tuple(t.shape)}"
        pts = torch.cat((normalize_aabb(xyz, self.aabb), t), dim=-1)
        features = interpolate_ms_features(pts.reshape(-1, 4), self.grids,
                                           concat_features=True, num_levels=None)
        if features.numel() == 0:
            features = torch.zeros((0, 1), device=xyz.device, dtype=xyz.dtype)
        return features

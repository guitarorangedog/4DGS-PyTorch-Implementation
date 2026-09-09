"""Spatial-temporal HexPlane regularization [4D].

Traced from the official live path (NOT from memory):

- ``train.py`` (fine stage only, gated on ``time_smoothness_weight != 0``)::

      tv_loss = gaussians.compute_regulation(
          hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)

- ``scene/gaussian_model.py::compute_regulation`` (argument ORDER matters)::

      plane_tv_weight * _plane_regulation()
      + time_smoothness_weight * _time_regulation()
      + l1_time_planes_weight * _l1_regulation()

NAMING TRAP (verified by grep): ``_plane_regulation`` does NOT call
``compute_plane_tv``. The first-order ``compute_plane_tv`` lives only in the
dead ``Regularizer`` classes of ``scene/regulation.py`` (never instantiated
anywhere). The live spatial term uses the SECOND-order
``compute_plane_smoothness``, same as the temporal term. It is ported below
as :func:`plane_total_variation` for completeness and marked non-live.

Live terms (all differentiate w.r.t. HexPlane grids ONLY):

- spatial: planes ``[0, 1, 3]`` = ``xy, xz, yz``, second-difference
  smoothness, weight ``plane_tv_weight = 1e-4``.
- temporal: planes ``[2, 4, 5]`` = ``xt, yt, zt``, second-difference
  smoothness, weight ``time_smoothness_weight = 1e-2``.
- L1: planes ``[2, 4, 5]``, ``mean(|1 - grid|)``, weight
  ``l1_time_planes = 1e-4``. The reference is ONE, matching the identity
  initialization of time planes (``nn.init.ones_``): untrained temporal
  modulation stays multiplicatively neutral, and the penalty only prices
  deviation from neutrality.

Aggregation: plain SUM over resolution levels and selected planes (official
``total = 0`` accumulator). The ``len(grids) == 3`` guards (3-plane KPlanes
compat) are preserved but dead for our 6-plane levels.
"""

from dataclasses import dataclass

import torch

from deformation.hexplane import TIME_PLANE_IDS

__all__ = [
    "SPATIAL_PLANE_IDS",
    "TIME_PLANE_IDS",
    "RegWeights",
    "RegTerms",
    "plane_total_variation",
    "second_difference_smoothness",
    "spatial_smoothness_term",
    "temporal_smoothness_term",
    "l1_time_planes_term",
    "regularization_terms",
    "weighted_regularization",
]

#: Official spatial plane ids (xy, xz, yz) for the "plane TV" weight.
SPATIAL_PLANE_IDS = (0, 1, 3)


@dataclass
class RegWeights:
    """Official ``ModelHiddenParams`` regularization defaults (D-NeRF)."""

    plane_tv: float = 0.0001  #: spatial second-difference smoothness.
    time_smoothness: float = 0.01  #: temporal second-difference smoothness.
    l1_time_planes: float = 0.0001  #: |1 - grid| on temporal planes.


@dataclass
class RegTerms:
    """Unweighted per-term scalars (all differentiable w.r.t. grids)."""

    spatial: torch.Tensor
    temporal: torch.Tensor
    l1_time: torch.Tensor


def plane_total_variation(t: torch.Tensor) -> torch.Tensor:
    """First-order plane TV (official ``compute_plane_tv``, NON-live path).

    ``2 * (mean_h + mean_w)`` of squared neighbor differences, each direction
    normalized by its element count. Ported verbatim; the live model path
    never calls it (see module docstring).
    """
    _, _, h, w = t.shape
    count_h = t.shape[0] * t.shape[1] * (h - 1) * w
    count_w = t.shape[0] * t.shape[1] * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


def second_difference_smoothness(t: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness along the height dim (official live term).

    ``mean((t[i+1] - 2*t[i] + t[i-1])^2)`` over ``i = 1..h-2``. Zero for
    constant AND linear profiles; penalizes curvature. Used for BOTH the
    spatial (``xy/xz/yz``) and temporal (``xt/yt/zt``) live terms.
    """
    first = t[..., 1:, :] - t[..., :-1, :]
    second = first[..., 1:, :] - first[..., :-1, :]
    return torch.square(second).mean()


def _sum_over_planes(levels, plane_ids) -> torch.Tensor:
    total = 0.0
    for level in levels:
        ids = () if len(level) == 3 else plane_ids
        for i in ids:
            total = total + second_difference_smoothness(level[i])
    return total if isinstance(total, torch.Tensor) else torch.as_tensor(total)


def spatial_smoothness_term(levels) -> torch.Tensor:
    """Live "plane TV" term: second-difference over ``xy/xz/yz`` (all levels)."""
    return _sum_over_planes(levels, SPATIAL_PLANE_IDS)


def temporal_smoothness_term(levels) -> torch.Tensor:
    """Live temporal term: second-difference over ``xt/yt/zt`` (all levels)."""
    return _sum_over_planes(levels, TIME_PLANE_IDS)


def l1_time_planes_term(levels) -> torch.Tensor:
    """Live L1 term: ``mean(|1 - grid|)`` over ``xt/yt/zt`` (all levels)."""
    total = 0.0
    for level in levels:
        if len(level) == 3:
            continue
        for i in TIME_PLANE_IDS:
            total = total + torch.abs(1 - level[i]).mean()
    return total if isinstance(total, torch.Tensor) else torch.as_tensor(total)


def regularization_terms(field) -> RegTerms:
    """Unweighted ``(spatial, temporal, l1_time)`` terms for a field."""
    levels = field.hexplane.grids
    return RegTerms(
        spatial=spatial_smoothness_term(levels),
        temporal=temporal_smoothness_term(levels),
        l1_time=l1_time_planes_term(levels),
    )


def weighted_regularization(field, weights: RegWeights | None = None) -> torch.Tensor:
    """``w_plane * spatial + w_time * temporal + w_l1 * l1_time``.

    The scalar the Commit 16 trainer adds to the fine-stage loss (official
    gating: fine stage AND ``time_smoothness != 0`` — enforced by the caller).
    """
    if weights is None:
        weights = RegWeights()
    terms = regularization_terms(field)
    return (
        weights.plane_tv * terms.spatial
        + weights.time_smoothness * terms.temporal
        + weights.l1_time_planes * terms.l1_time
    )

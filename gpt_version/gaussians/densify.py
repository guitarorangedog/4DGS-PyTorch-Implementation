"""Adaptive Gaussian population control inherited from 3D-GS [STATIC].

Mirrors ``scene/gaussian_model.py`` of 3D-GS / 4DGaussians
(``densify_and_clone``, ``densify_and_split``, ``prune_points``/``prune``,
``reset_opacity``, ``densification_postfix``, ``add_densification_stats``)
with two documented deviations: everything is device-agnostic (official code
hardcodes ``device="cuda"`` and calls ``torch.cuda.empty_cache()``), and the
4D-specific ``_deformation_table`` / ``_deformation_accum`` bookkeeping does
not exist yet (deformation arrives in Commits 11+).

Why optimizer-state surgery is unavoidable
------------------------------------------
``torch.optim.Adam`` holds Python references to the exact ``nn.Parameter``
objects passed at construction, plus per-parameter momentum buffers
``exp_avg`` / ``exp_avg_sq`` shaped ``[N, ...]``. Cloning, splitting and
pruning change ``N``, which requires brand-new tensors; the old
``nn.Parameter`` objects cannot change shape in place. Every population op
must therefore (a) build the new concatenated/sliced tensor, (b) wrap it in
a fresh ``nn.Parameter`` and repoint the optimizer group at it, and (c) grow
(zero-fill), slice, or reset the momentum buffers in lockstep — otherwise the
buffer leading dimension desyncs from the parameter and the next
``optimizer.step()`` fails or silently corrupts. The helpers
:func:`_append_to_optimizer`, :func:`_prune_optimizer` and
:func:`_replace_tensor_in_optimizer` implement exactly this, mirroring the
official ``cat_tensors_to_optimizer`` / ``_prune_optimizer`` /
``replace_tensor_to_optimizer``.

Group names (explicit contract for the Commit 8 trainer):
``"xyz"``, ``"f_dc"``, ``"f_rest"``, ``"opacity"``, ``"scaling"``,
``"rotation"`` — identical to official ``training_setup`` minus the
``"deformation"`` / ``"grid"`` groups, which do not exist yet.
"""

import torch
from torch import nn

from gaussians.geometry import build_rotation
from gaussians.gaussian_model import inverse_sigmoid

__all__ = [
    "OPTIM_GROUP_NAMES",
    "attach_optimizer",
    "init_densify_stats",
    "add_densification_stats",
    "mean_viewspace_grads",
    "select_clone_mask",
    "select_split_mask",
    "densify_and_clone",
    "densify_and_split",
    "densify",
    "prune_points",
    "prune_by_opacity_and_size",
    "reset_opacity",
]

OPTIM_GROUP_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

#: Maps optimizer group name -> CanonicalGaussianModel attribute name.
_GROUP_TO_ATTR = {
    "xyz": "_xyz",
    "f_dc": "_features_dc",
    "f_rest": "_features_rest",
    "opacity": "_opacity",
    "scaling": "_scaling",
    "rotation": "_rotation",
}

OPACITY_RESET_CAP = 0.01


# -- minimal optimizer / stats setup (full LRs + schedulers: Commit 8) --------
def attach_optimizer(model, lr: float = 1e-3) -> torch.optim.Adam:
    """Attach a minimal single-LR Adam with the six official group names.

    Minimal test helper retained from Commit 5. The canonical training path
    is ``training/optim.setup_static_optimizer`` (Commit 8), which assigns
    the official per-group LRs; both go through the same generic surgery
    helpers below, so population control works identically either way.

    Args:
        model: :class:`CanonicalGaussianModel` with initialized parameters.
        lr: learning rate for all six groups.

    Returns:
        The created ``torch.optim.Adam`` (also stored as ``model.optimizer``).
    """
    groups = [
        {"params": [getattr(model, _GROUP_TO_ATTR[name])], "lr": lr, "name": name}
        for name in OPTIM_GROUP_NAMES
    ]
    model.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    return model.optimizer


def init_densify_stats(model) -> None:
    """Allocate zeroed densification buffers sized to current ``N``.

    Creates ``xyz_gradient_accum: [N, 1]``, ``denom: [N, 1]`` and
    ``max_radii2D: [N]`` on the model's device (official ``training_setup``
    minus the 4D accumulators).
    """
    device = model._xyz.device
    n = model.num_points
    model.xyz_gradient_accum = torch.zeros((n, 1), dtype=torch.float32, device=device)
    model.denom = torch.zeros((n, 1), dtype=torch.float32, device=device)
    model.max_radii2D = torch.zeros((n,), dtype=torch.float32, device=device)


# -- statistics ----------------------------------------------------------------
def add_densification_stats(model, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor) -> None:
    """Accumulate 2D view-space gradient norms (official ``add_densification_stats``).

    Args:
        model: model carrying ``xyz_gradient_accum: [N, 1]`` / ``denom: [N, 1]``.
        viewspace_point_tensor: ``[M, D>=2]`` screen-space point gradients
            (only the ``:2`` image-plane components are used, as in official).
        update_filter: ``[M]`` bool mask of visible Gaussians this iteration.
    """
    grad_norm = torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
    model.xyz_gradient_accum[update_filter] += grad_norm
    model.denom[update_filter] += 1


def mean_viewspace_grads(model) -> torch.Tensor:
    """Mean gradient per Gaussian: ``accum / denom`` with ``NaN -> 0``.

    Returns:
        ``[N, 1]`` (official ``densify`` pre-processing).
    """
    grads = model.xyz_gradient_accum / model.denom
    grads[grads.isnan()] = 0.0
    return grads


# -- selection rules ------------------------------------------------------------
def select_clone_mask(grads: torch.Tensor, max_scales: torch.Tensor,
                      grad_threshold: float, percent_dense: float, scene_extent: float) -> torch.Tensor:
    """Clone mask: high gradient AND spatially small (official ``densify_and_clone``).

    Args:
        grads: ``[N, 1]`` mean view-space grads.
        max_scales: ``[N]`` max activated scale per Gaussian.
        grad_threshold: minimum gradient norm.
        percent_dense: fraction of ``scene_extent`` below which a Gaussian
            counts as "small" (official default ``0.01``).
        scene_extent: scene radius from ``getNerfppNorm``-style normalization.

    Returns:
        ``[N]`` bool mask.
    """
    high_grad = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    small = max_scales <= percent_dense * scene_extent
    return torch.logical_and(high_grad, small)


def select_split_mask(n_current: int, grads: torch.Tensor, max_scales: torch.Tensor,
                      grad_threshold: float, percent_dense: float, scene_extent: float) -> torch.Tensor:
    """Split mask: high gradient AND spatially large (official ``densify_and_split``).

    ``grads`` may be shorter than ``n_current`` (it predates a clone step in
    the same ``densify`` call), so it is zero-padded to ``n_current`` exactly
    like the official ``padded_grad``.

    Returns:
        ``[n_current]`` bool mask.
    """
    padded = torch.zeros((n_current,), dtype=grads.dtype, device=grads.device)
    padded[: grads.shape[0]] = grads.squeeze()
    high_grad = torch.where(padded >= grad_threshold, True, False)
    large = max_scales > percent_dense * scene_extent
    return torch.logical_and(high_grad, large)


# -- optimizer surgery -----------------------------------------------------------
def _single_param_groups(model):
    for group in model.optimizer.param_groups:
        if len(group["params"]) == 1:
            yield group


def _append_to_optimizer(model, extension: dict[str, torch.Tensor]) -> dict[str, nn.Parameter]:
    """Append new rows to every optimizer param + zero-fill momentum.

    Mirrors official ``cat_tensors_to_optimizer``.
    """
    out: dict[str, nn.Parameter] = {}
    for group in _single_param_groups(model):
        ext = extension[group["name"]]
        old = group["params"][0]
        state = model.optimizer.state.get(old, None)
        new_param = nn.Parameter(torch.cat((old, ext), dim=0).requires_grad_(True))
        if state is not None:
            state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(ext)), dim=0)
            state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(ext)), dim=0)
            del model.optimizer.state[old]
            model.optimizer.state[new_param] = state
        group["params"][0] = new_param
        out[group["name"]] = new_param
    return out


def _prune_optimizer(model, keep_mask: torch.Tensor) -> dict[str, nn.Parameter]:
    """Slice every optimizer param and its momentum buffers to ``keep_mask``.

    Mirrors official ``_prune_optimizer`` (official takes the keep mask under
    the name ``mask`` and computes ``valid_points_mask = ~mask`` one level up).
    """
    out: dict[str, nn.Parameter] = {}
    for group in _single_param_groups(model):
        old = group["params"][0]
        state = model.optimizer.state.get(old, None)
        new_param = nn.Parameter(old[keep_mask].requires_grad_(True))
        if state is not None:
            state["exp_avg"] = state["exp_avg"][keep_mask]
            state["exp_avg_sq"] = state["exp_avg_sq"][keep_mask]
            del model.optimizer.state[old]
            model.optimizer.state[new_param] = state
        group["params"][0] = new_param
        out[group["name"]] = new_param
    return out


def _replace_tensor_in_optimizer(model, tensor: torch.Tensor, name: str) -> dict[str, nn.Parameter]:
    """Swap one group's tensor in place, resetting its momentum to zeros.

    Mirrors official ``replace_tensor_to_optimizer`` (used by opacity reset).
    """
    out: dict[str, nn.Parameter] = {}
    for group in model.optimizer.param_groups:
        if group["name"] == name:
            old = group["params"][0]
            state = model.optimizer.state.get(old, None)
            new_param = nn.Parameter(tensor.requires_grad_(True))
            if state is not None:
                state["exp_avg"] = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)
                del model.optimizer.state[old]
                model.optimizer.state[new_param] = state
            group["params"][0] = new_param
            out[group["name"]] = new_param
    return out


def _sync_params_from_optimizer(model, optimizable: dict[str, nn.Parameter]) -> None:
    model._xyz = optimizable["xyz"]
    model._features_dc = optimizable["f_dc"]
    model._features_rest = optimizable["f_rest"]
    model._opacity = optimizable["opacity"]
    model._scaling = optimizable["scaling"]
    model._rotation = optimizable["rotation"]


def _reset_stats(model) -> None:
    """Zero all densification buffers for the current ``N``.

    Mirrors official ``densification_postfix`` tail: after any append, every
    accumulator is discarded and re-zeroed at the new size.
    """
    init_densify_stats(model)


# -- population ops ---------------------------------------------------------------
def prune_points(model, prune_mask: torch.Tensor) -> None:
    """Remove masked Gaussians from every parameter, stat and optimizer slot.

    Args:
        prune_mask: ``[N]`` bool, ``True`` = remove (official ``prune_points``
            receives this mask and internally negates it for slicing).
    """
    keep = ~prune_mask
    _sync_params_from_optimizer(model, _prune_optimizer(model, keep))
    model.xyz_gradient_accum = model.xyz_gradient_accum[keep]
    model.denom = model.denom[keep]
    model.max_radii2D = model.max_radii2D[keep]


def densification_postfix(model, new_xyz: torch.Tensor, new_features_dc: torch.Tensor,
                          new_features_rest: torch.Tensor, new_opacities: torch.Tensor,
                          new_scaling: torch.Tensor, new_rotation: torch.Tensor) -> None:
    """Append children and reset stats (official ``densification_postfix``).

    Takes **raw** parameter values (log-scales, quats, logits, SH), matching
    the official call sites which pass raw tensors through.
    """
    _sync_params_from_optimizer(model, _append_to_optimizer(model, {
        "xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling": new_scaling,
        "rotation": new_rotation,
    }))
    _reset_stats(model)


def densify_and_clone(model, grads: torch.Tensor, grad_threshold: float,
                      scene_extent: float, percent_dense: float = 0.01) -> int:
    """Duplicate small high-gradient Gaussians in place (parents kept).

    Children copy the parent's **raw** parameters unchanged (no shrink).

    Returns:
        Number of Gaussians cloned.
    """
    mask = select_clone_mask(grads, model.get_scaling.max(dim=1).values,
                             grad_threshold, percent_dense, scene_extent)
    if not mask.any():
        return 0
    densification_postfix(
        model,
        model._xyz[mask],
        model._features_dc[mask],
        model._features_rest[mask],
        model._opacity[mask],
        model._scaling[mask],
        model._rotation[mask],
    )
    return int(mask.sum())


def densify_and_split(model, grads: torch.Tensor, grad_threshold: float,
                      scene_extent: float, percent_dense: float = 0.01,
                      num_children: int = 2) -> int:
    """Split large high-gradient Gaussians into ``num_children`` (default 2).

    Per parent: sample ``num_children`` offsets from ``N(0, diag(s))`` in
    local frame, rotate by the parent rotation and translate to the parent
    position; child log-scales shrink by ``log(parent_scale / (0.8 * N))``;
    remaining raw attributes are copied. Parents are then pruned, so ``N``
    grows by ``(num_children - 1) * M`` for ``M`` selected parents.

    Returns:
        Number of parent Gaussians split.
    """
    mask = select_split_mask(model.num_points, grads, model.get_scaling.max(dim=1).values,
                             grad_threshold, percent_dense, scene_extent)
    if not mask.any():
        return 0
    n_children_total = int(mask.sum()) * num_children
    stds = model.get_scaling[mask].repeat(num_children, 1)
    samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
    rots = build_rotation(model._rotation[mask]).repeat(num_children, 1, 1)
    new_xyz = (
        torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        + model.get_xyz[mask].repeat(num_children, 1)
    )
    new_scaling = torch.log(
        model.get_scaling[mask].repeat(num_children, 1) / (0.8 * num_children)
    )
    densification_postfix(
        model,
        new_xyz,
        model._features_dc[mask].repeat(num_children, 1, 1),
        model._features_rest[mask].repeat(num_children, 1, 1),
        model._opacity[mask].repeat(num_children, 1),
        new_scaling,
        model._rotation[mask].repeat(num_children, 1),
    )
    prune_filter = torch.cat((
        mask,
        torch.zeros(n_children_total, dtype=torch.bool, device=mask.device),
    ))
    prune_points(model, prune_filter)
    return int(mask.sum())


def densify(model, grad_threshold: float, scene_extent: float,
            percent_dense: float = 0.01, num_children: int = 2) -> tuple[int, int]:
    """Full densification pass: clone small, then split large (official ``densify``).

    Gradients are computed once from the accumulators; the split step
    zero-pads them to the post-clone ``N`` exactly like the official code.

    Returns:
        ``(n_cloned, n_split_parents)``.
    """
    grads = mean_viewspace_grads(model)
    n_cloned = densify_and_clone(model, grads, grad_threshold, scene_extent, percent_dense)
    n_split = densify_and_split(model, grads, grad_threshold, scene_extent, percent_dense, num_children)
    return n_cloned, n_split


def prune_by_opacity_and_size(model, min_opacity: float, scene_extent: float,
                              max_screen_size: float | None = None) -> int:
    """Prune transparent and (optionally) oversized Gaussians.

    Mirrors official ``prune(max_grad [unused], min_opacity, extent,
    max_screen_size)``. Effective mask (note the official code ORs the
    screen-size term twice, which is redundant but harmless — reproduced
    here as the simplified equivalent)::

        opacity_low | big_vs | big_ws

    where ``big_vs = max_radii2D > max_screen_size`` and
    ``big_ws = max_scale > 0.1 * extent``. No CUDA cache flush on CPU.

    Returns:
        Number of Gaussians removed.
    """
    prune_mask = (model.get_opacity < min_opacity).squeeze(-1)
    if max_screen_size:
        big_vs = model.max_radii2D > max_screen_size
        big_ws = model.get_scaling.max(dim=1).values > 0.1 * scene_extent
        prune_mask = prune_mask | big_vs | big_ws
    if not prune_mask.any():
        return 0
    prune_points(model, prune_mask)
    return int(prune_mask.sum())


@torch.no_grad()
def reset_opacity(model, cap: float = OPACITY_RESET_CAP) -> None:
    """Cap opacities at ``cap`` (default ``0.01``) with optimizer consistency.

    Mirrors official ``reset_opacity``: ``logit(min(opacity, cap))`` replaces
    the opacity tensor and its Adam momentum is zeroed (see
    :func:`_replace_tensor_in_optimizer`). Note the ``@torch.no_grad``
    decorator matches official semantics (not tracked by autograd); callers
    must not expect gradients through the reset.
    """
    new_opacities = inverse_sigmoid(torch.min(model.get_opacity, torch.full_like(model.get_opacity, cap)))
    model._opacity = _replace_tensor_in_optimizer(model, new_opacities, "opacity")["opacity"]

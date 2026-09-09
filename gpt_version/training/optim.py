"""Static optimizer setup: named Adam groups + xyz scheduler [INFRA].

Group names are the explicit contract shared with Commit 5 densification
surgery (``gaussians/densify.py``): ``xyz``, ``f_dc``, ``f_rest``,
``opacity``, ``scaling``, ``rotation`` — identical to official
``GaussianModel.training_setup`` minus the ``deformation``/``grid`` groups,
which do not exist until 4D training (Commit 16).

Learning rates match official ``arguments.OptimizationParams`` defaults:

- ``xyz``: ``position_lr_init * spatial_lr_scale``
- ``f_dc``: ``feature_lr``; ``f_rest``: ``feature_lr / 20``
- ``opacity`` / ``scaling`` / ``rotation``: fixed LRs below.

This module only *constructs* the groups; the append/prune/replace surgery
itself lives in ``gaussians/densify.py`` and operates generically over
``model.optimizer.param_groups``, so no surgery logic is duplicated here.
"""

from dataclasses import dataclass

import torch

from gaussians.densify import OPTIM_GROUP_NAMES, init_densify_stats
from training.schedules import get_expon_lr_func

__all__ = [
    "StaticTrainingConfig",
    "DeformationOptimConfig",
    "setup_static_optimizer",
    "setup_4d_optimizer",
    "get_group",
    "update_xyz_lr",
    "update_4d_lrs",
]


@dataclass
class StaticTrainingConfig:
    """Official static-training hyperparameters (4DGaussians defaults)."""

    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 20_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.0  # official 4DGS default (pure L1); 3D-GS used 0.2


@dataclass
class DeformationOptimConfig:
    """Official D-NeRF deformation/grid LR hyperparameters.

    Traced from ``arguments/dnerf/dnerf_default.py`` (which OVERRIDES the
    global ``arguments/__init__.py`` defaults): D-NeRF uses finals
    ``deformation 1.6e-4 -> 1.6e-6`` and ``grid 1.6e-3 -> 1.6e-5``
    (global defaults are 1.6e-5 and 1.6e-4 respectively — do not confuse
    them). All three schedules share the xyz horizon
    (``position_lr_max_steps``) and the deformation delay multiplier.
    """

    deformation_lr_init: float = 0.00016
    deformation_lr_final: float = 0.0000016
    deformation_lr_delay_mult: float = 0.01
    grid_lr_init: float = 0.0016
    grid_lr_final: float = 0.000016
    lr_max_steps: int = 20_000


def _static_groups(model, cfg: StaticTrainingConfig, spatial_lr_scale: float) -> list[dict]:
    params = dict(model.static_param_groups())
    assert tuple(params) == OPTIM_GROUP_NAMES, tuple(params)
    lrs = {
        "xyz": cfg.position_lr_init * spatial_lr_scale,
        "f_dc": cfg.feature_lr,
        "f_rest": cfg.feature_lr / 20.0,
        "opacity": cfg.opacity_lr,
        "scaling": cfg.scaling_lr,
        "rotation": cfg.rotation_lr,
    }
    return [{"params": [params[name]], "lr": lrs[name], "name": name}
            for name in OPTIM_GROUP_NAMES]


def setup_static_optimizer(model, cfg: StaticTrainingConfig | None = None,
                           spatial_lr_scale: float = 1.0) -> torch.optim.Adam:
    """Create the six named Adam groups on the model's parameters.

    Mirrors official ``training_setup`` (static subset): one group per
    parameter with the LRs above, ``Adam(lr=0.0, eps=1e-15)`` base, plus the
    exponential xyz schedule stored as ``model.xyz_schedule``. Also records
    ``model.spatial_lr_scale`` / ``model.percent_dense`` and allocates the
    densification buffers via :func:`init_densify_stats`.

    Args:
        model: :class:`CanonicalGaussianModel` with initialized parameters.
        cfg: hyperparameter overrides (defaults = official values).
        spatial_lr_scale: scene-extent scale for the position LR
            (official ``self.spatial_lr_scale``, set from the scene radius).
    """
    if cfg is None:
        cfg = StaticTrainingConfig()
    model.optimizer = torch.optim.Adam(
        _static_groups(model, cfg, spatial_lr_scale), lr=0.0, eps=1e-15)
    model.spatial_lr_scale = spatial_lr_scale
    model.percent_dense = cfg.percent_dense
    model.xyz_schedule = get_expon_lr_func(
        lr_init=cfg.position_lr_init * spatial_lr_scale,
        lr_final=cfg.position_lr_final * spatial_lr_scale,
        lr_delay_mult=cfg.position_lr_delay_mult,
        max_steps=cfg.position_lr_max_steps,
    )
    init_densify_stats(model)
    return model.optimizer


def setup_4d_optimizer(model, field, static_cfg: StaticTrainingConfig | None = None,
                       deform_cfg: DeformationOptimConfig | None = None,
                       spatial_lr_scale: float = 1.0) -> torch.optim.Adam:
    """Create the full eight-group 4D optimizer (official ``training_setup``).

    Six single-parameter Gaussian groups (identical to
    :func:`setup_static_optimizer`) PLUS two multi-parameter groups::

        "deformation": decoder (+ aux) MLP params, LR ``deform_init * scale``
        "grid": HexPlane grid params, LR ``grid_init * scale``

    Multi-parameter groups are intentionally skipped by the Commit 5
    append/prune surgery (mirroring official ``_prune_optimizer``, which
    skips ``len(params) > 1`` groups): densification only reshapes
    per-Gaussian tensors. Schedules for xyz/deformation/grid are stored as
    ``model.xyz_schedule`` / ``model.deform_schedule`` / ``model.grid_schedule``.

    Called fresh at EACH stage (official ``training_setup`` per
    ``scene_reconstruction``): fresh Adam state + zeroed densify stats.
    """
    if static_cfg is None:
        static_cfg = StaticTrainingConfig()
    if deform_cfg is None:
        deform_cfg = DeformationOptimConfig()
    groups = _static_groups(model, static_cfg, spatial_lr_scale)
    groups.append({"params": list(field.get_mlp_parameters()),
                   "lr": deform_cfg.deformation_lr_init * spatial_lr_scale,
                   "name": "deformation"})
    groups.append({"params": list(field.get_grid_parameters()),
                   "lr": deform_cfg.grid_lr_init * spatial_lr_scale,
                   "name": "grid"})
    model.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    model.spatial_lr_scale = spatial_lr_scale
    model.percent_dense = static_cfg.percent_dense
    model.xyz_schedule = get_expon_lr_func(
        lr_init=static_cfg.position_lr_init * spatial_lr_scale,
        lr_final=static_cfg.position_lr_final * spatial_lr_scale,
        lr_delay_mult=static_cfg.position_lr_delay_mult,
        max_steps=static_cfg.position_lr_max_steps,
    )
    model.deform_schedule = get_expon_lr_func(
        lr_init=deform_cfg.deformation_lr_init * spatial_lr_scale,
        lr_final=deform_cfg.deformation_lr_final * spatial_lr_scale,
        lr_delay_mult=deform_cfg.deformation_lr_delay_mult,
        max_steps=deform_cfg.lr_max_steps,
    )
    model.grid_schedule = get_expon_lr_func(
        lr_init=deform_cfg.grid_lr_init * spatial_lr_scale,
        lr_final=deform_cfg.grid_lr_final * spatial_lr_scale,
        lr_delay_mult=deform_cfg.deformation_lr_delay_mult,
        max_steps=deform_cfg.lr_max_steps,
    )
    init_densify_stats(model)
    return model.optimizer


def get_group(model, name: str) -> dict:
    """Return the optimizer param group dict with the given name."""
    for group in model.optimizer.param_groups:
        if group["name"] == name:
            return group
    raise KeyError(f"no optimizer group named {name!r}")


def update_xyz_lr(model, iteration: int) -> float:
    """Step the xyz schedule; all other groups keep constant LRs.

    Static-trainer path (Commit 8). The 4D path uses :func:`update_4d_lrs`.

    Returns:
        The new xyz learning rate.
    """
    lr = model.xyz_schedule(iteration)
    get_group(model, "xyz")["lr"] = lr
    return lr


def update_4d_lrs(model, iteration: int) -> dict[str, float]:
    """Step xyz + deformation + grid schedules (official ``update_learning_rate``).

    Mirrors the official branch structure (``xyz`` by name, ``"grid" in name``,
    ``deformation`` by name); all other groups keep constant LRs.

    Returns:
        ``{"xyz": lr, "deformation": lr, "grid": lr}``.
    """
    lrs = {
        "xyz": model.xyz_schedule(iteration),
        "deformation": model.deform_schedule(iteration),
        "grid": model.grid_schedule(iteration),
    }
    for group in model.optimizer.param_groups:
        if group["name"] == "xyz":
            group["lr"] = lrs["xyz"]
        if "grid" in group["name"]:
            group["lr"] = lrs["grid"]
        elif group["name"] == "deformation":
            group["lr"] = lrs["deformation"]
    return lrs

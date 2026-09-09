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
    "setup_static_optimizer",
    "get_group",
    "update_xyz_lr",
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
    lrs = {
        "xyz": cfg.position_lr_init * spatial_lr_scale,
        "f_dc": cfg.feature_lr,
        "f_rest": cfg.feature_lr / 20.0,
        "opacity": cfg.opacity_lr,
        "scaling": cfg.scaling_lr,
        "rotation": cfg.rotation_lr,
    }
    params = dict(model.static_param_groups())
    assert tuple(params) == OPTIM_GROUP_NAMES, tuple(params)
    groups = [{"params": [params[name]], "lr": lrs[name], "name": name}
              for name in OPTIM_GROUP_NAMES]
    model.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
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


def get_group(model, name: str) -> dict:
    """Return the optimizer param group dict with the given name."""
    for group in model.optimizer.param_groups:
        if group["name"] == name:
            return group
    raise KeyError(f"no optimizer group named {name!r}")


def update_xyz_lr(model, iteration: int) -> float:
    """Step the xyz schedule; all other groups keep constant LRs.

    Mirrors the ``"xyz"`` branch of official ``update_learning_rate``
    (deformation/grid branches arrive in Commit 16).

    Returns:
        The new xyz learning rate.
    """
    lr = model.xyz_schedule(iteration)
    get_group(model, "xyz")["lr"] = lr
    return lr

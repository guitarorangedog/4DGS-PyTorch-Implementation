"""HyperNeRF training configuration.

Traced from ``arguments/hypernerf/default.py`` over global defaults.
Meaningful differences from D-NeRF:

- HexPlane: ``output_coordinate_dim = 16``, temporal res 150,
  ``multires = [1, 2, 4]`` (wider than D-NeRF/DyNeRF ``[1, 2]``),
  ``net_width = 128``, ``defor_depth = 1`` (single-Linear trunk, same as
  ``defor_depth = 0`` — see ``DecoderConfig`` mapping note).
- Regularization: ``plane_tv 2e-4 / time_smoothness 1e-3 / l1 1e-4``.
- Schedule: fine ``iterations = 14000``, ``densify_until_iter = 10000``,
  ``opacity_reset_interval = 300000``; grid LRs / thresholds / pruning
  interval commented out -> GLOBAL values (incl. pruning 100).
- LR inits/finals NOT overridden -> GLOBAL values apply.
- Opacity/SH deformation stays disabled (global ``no_do/no_dshs``).
"""

from dataclasses import replace

from deformation.decoder import DecoderConfig
from deformation.field import FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.regularize import RegWeights
from training.optim import DeformationOptimConfig
from training.trainer_4d import FourDTrainerConfig

__all__ = ["hypernerf_config"]


def hypernerf_config(**overrides) -> FourDTrainerConfig:
    """Faithful HyperNeRF config."""
    cfg = FourDTrainerConfig(
        fine_iterations=14_000,
        densify_until_iter=10_000,
        opacity_reset_interval=300_000,
        pruning_interval=100,
        reg=RegWeights(plane_tv=0.0002, time_smoothness=0.001,
                       l1_time_planes=0.0001),
        deform_optim=DeformationOptimConfig(
            deformation_lr_final=0.000016, grid_lr_final=0.00016),
        field=FieldConfig(
            hexplane=HexPlaneConfig(output_coordinate_dim=16,
                                    resolution=[64, 64, 64, 150],
                                    multires=[1, 2, 4]),
            decoder=DecoderConfig(width=128, depth=1),
            enable_aux=False),
    )
    return replace(cfg, **overrides) if overrides else cfg

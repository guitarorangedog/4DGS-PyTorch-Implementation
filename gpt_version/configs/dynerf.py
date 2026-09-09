"""DyNeRF / Neural-3D-Video training configuration.

Traced from ``arguments/dynerf/default.py`` over global defaults. Meaningful
differences from D-NeRF:

- HexPlane: ``output_coordinate_dim = 16`` (not 32), temporal res 150
  (not 25), ``multires = [1, 2]`` (same as D-NeRF), ``net_width = 128``.
- Regularization: ``plane_tv 2e-4 / time_smoothness 1e-3 / l1 1e-4``.
- Schedule: fine ``iterations = 14000``, ``densify_until_iter = 10000``,
  ``opacity_reset_interval = 60000``, ``pruning_interval`` NOT overridden
  (global 100 applies).
- LR inits/finals NOT overridden -> GLOBAL values apply
  (deformation 1.6e-4 -> 1.6e-5, grid 1.6e-3 -> 1.6e-4).
- ``no_do = False, no_dshs = False``: opacity + SH deform ENABLED
  (wired to ``FieldConfig.enable_aux``); D-NeRF disables both.
- ``dataloader/batch_size = 4``: official multi-view batching is NOT
  reproduced (single-view sampling; documented limitation).
- Working resolution 1352x1014, black background (data layer).
"""

from dataclasses import replace

from deformation.decoder import DecoderConfig
from deformation.field import FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.regularize import RegWeights
from training.optim import DeformationOptimConfig
from training.trainer_4d import FourDTrainerConfig

__all__ = ["dynerf_config"]


def dynerf_config(**overrides) -> FourDTrainerConfig:
    """Faithful DyNeRF config."""
    cfg = FourDTrainerConfig(
        fine_iterations=14_000,
        densify_until_iter=10_000,
        opacity_reset_interval=60_000,
        pruning_interval=100,
        reg=RegWeights(plane_tv=0.0002, time_smoothness=0.001,
                       l1_time_planes=0.0001),
        deform_optim=DeformationOptimConfig(
            deformation_lr_final=0.000016, grid_lr_final=0.00016),
        field=FieldConfig(
            hexplane=HexPlaneConfig(output_coordinate_dim=16,
                                    resolution=[64, 64, 64, 150],
                                    multires=[1, 2]),
            decoder=DecoderConfig(width=128, depth=1),
            enable_aux=True),
    )
    return replace(cfg, **overrides) if overrides else cfg

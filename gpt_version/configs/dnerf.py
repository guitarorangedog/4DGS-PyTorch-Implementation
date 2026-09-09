"""D-NeRF training configuration (reference).

Traced from ``arguments/dnerf/dnerf_default.py`` over global defaults::

    coarse_iterations = 3000, fine iterations = 20000
    deformation 1.6e-4 -> 1.6e-6, grid 1.6e-3 -> 1.6e-5 (D-NeRF overrides!)
    pruning_interval = 8000 (D-NeRF override)
    multires = [1, 2], defor_depth = 0 (-> our decoder depth 1),
    net_width = 64, plane_tv 1e-4, time_smoothness 1e-2, l1 1e-4, bounds 1.6

``FourDTrainerConfig`` defaults already equal this file; the factory below
pins that fact (and guards it in tests).
"""

from training.trainer_4d import FourDTrainerConfig

__all__ = ["dnerf_config"]


def dnerf_config(**overrides) -> FourDTrainerConfig:
    """Faithful D-NeRF config (= :class:`FourDTrainerConfig` defaults)."""
    return FourDTrainerConfig(**overrides)

"""Per-dataset training configurations [INFRA].

Traced from official ``arguments/{dnerf,dynerf,hypernerf}/*default.py`` on top
of the global ``arguments/__init__.py`` defaults. Each factory returns a fully
populated :class:`FourDTrainerConfig`; only values the official files set
(or that our pipeline needs to mirror them) appear as overrides.

D-NeRF is the reference: ``FourDTrainerConfig`` defaults already equal it.
"""

from configs.dnerf import dnerf_config
from configs.dynerf import dynerf_config
from configs.hypernerf import hypernerf_config

__all__ = ["dnerf_config", "dynerf_config", "hypernerf_config", "get_trainer_config"]

_DATASETS = {
    "dnerf": dnerf_config,
    "dynerf": dynerf_config,
    "hypernerf": hypernerf_config,
}


def get_trainer_config(dataset_type: str):
    """Return the faithful training config for ``dnerf|dynerf|hypernerf``."""
    try:
        return _DATASETS[dataset_type]()
    except KeyError:
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r} "
            f"(expected one of {sorted(_DATASETS)}).")

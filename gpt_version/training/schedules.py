"""Learning-rate schedules matching 3D-GS conventions [INFRA].

``get_expon_lr_func`` is a verbatim port of
``utils/general_utils.get_expon_lr_func`` (Plenoxels/JaxNeRF style):
log-linear interpolation from ``lr_init`` (step 0) to ``lr_final``
(step ``max_steps``), optionally gated by a reverse-cosine delay that
starts the rate at ``lr_init * lr_delay_mult``.

Commit 8 scope: only the ``xyz`` position schedule is stepped during static
training (see ``training/optim.update_xyz_lr``). Deformation/grid schedules
arrive with 4D training (Commit 16).
"""

import numpy as np

__all__ = ["get_expon_lr_func"]


def get_expon_lr_func(lr_init: float, lr_final: float, lr_delay_steps: int = 0,
                      lr_delay_mult: float = 1.0, max_steps: int = 1_000_000):
    """Build a deterministic scalar LR schedule (official formula).

    Args:
        lr_init: rate at step 0 (before delay scaling).
        lr_final: rate at step ``max_steps``.
        lr_delay_steps: warm-up horizon; ``0`` disables the delay.
        lr_delay_mult: rate starts at ``lr_init * lr_delay_mult`` when
            ``lr_delay_steps > 0``, easing back to the normal curve.
        max_steps: step at which ``lr_final`` is reached (clamped beyond).

    Returns:
        Callable ``step -> float`` with ``f(0) = lr_init``
        (times ``lr_delay_mult`` if delayed) and ``f(max_steps) = lr_final``.
    """

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return float(delay_rate * log_lerp)

    return helper

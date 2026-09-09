"""Tiny deformation decoder: features -> (dX, ds, dr) [4D].

Paper Sec. 4.2, Eqs. 11-12: a compact MLP decodes the HexPlane voxel feature
``f_voxel`` into position / rotation / scaling deformations, applied as::

    (X', r', s') = (X + dX, r + dr, s + ds)

Faithful to the decoder slice of official ``scene/deformation.py``
(``Deformation.create_net`` + ``initialize_weights`` + the ``*_deform``
heads). The trunk consumes the grid feature ONLY (official ``query_time``
concatenates ``[grid_feature]``; the positional/time encodings computed in
``deform_network`` are unused on the default path).

Output semantics (all heads end in a Linear: UNBOUNDED):

- ``d_xyz [N, 3]``: additive world-space offset, ``X' = X + dX``.
- ``d_scaling [N, 3]``: additive delta in RAW LOG-scale space, i.e.
  ``_scaling' = _scaling + ds`` BEFORE the ``exp`` activation. (Official
  ``forward_dynamic`` adds ``ds`` to ``scales_emb[:, :3]``, which is the raw
  input slice of the positional encoding, and the renderer exponentiates.)
- ``d_rotation [N, 4]``: additive delta to the RAW quaternion,
  ``_rotation' = _rotation + dr``, normalized later by the model. (Official
  default ``apply_rotation=False``; the quaternion-multiply alternative in
  ``utils/graphics_utils.batch_quaternion_multiply`` is never used by default.)

``DeformationDecoder`` below is the paper/default path. The official module
also builds opacity (``[N, 1]``, raw-logit space) and SH (``[N, 16, 3]``)
heads, but the default config disables both (``no_do=True``,
``no_dshs=True``); they live in :class:`AuxDeformationHeads`, clearly
separated so the core stays legible.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    "DecoderConfig",
    "DeformationDeltas",
    "DeformationDecoder",
    "AuxDeformationHeads",
    "SH_COEFFS_PER_CHANNEL",
]

#: SH head width: degree-3 SH has 16 coeffs per channel (official ``16*3``).
SH_COEFFS_PER_CHANNEL = 16


@dataclass
class DecoderConfig:
    """Official ``ModelHiddenParams`` decoder defaults (``net_width = 64``).

    Depth mapping (naming difference, NOT architectural): official
    ``defor_depth`` counts EXTRA trunk layers (``range(D - 1)`` appended
    after the first Linear), so D-NeRF's ``defor_depth = 0`` builds a
    SINGLE-Linear trunk. Here ``depth`` counts TOTAL trunk Linears, so the
    faithful D-NeRF equivalent is ``depth = 1`` (the default).
    """

    width: int = 64  #: ``net_width``: trunk/head hidden dim.
    depth: int = 1  #: total trunk Linears (1 == official ``defor_depth = 0``).


@dataclass
class DeformationDeltas:
    """Core decoder outputs (all unbounded, raw-space additive)."""

    d_xyz: torch.Tensor  #: ``[N, 3]`` world-space offset.
    d_scaling: torch.Tensor  #: ``[N, 3]`` raw log-scale delta.
    d_rotation: torch.Tensor  #: ``[N, 4]`` raw quaternion delta.


def _head(width: int, out_dim: int) -> nn.Sequential:
    """One deformation head (official ``*_deform`` architecture)."""
    return nn.Sequential(
        nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, out_dim)
    )


def _init_linear_weights(module: nn.Module) -> None:
    """Official ``initialize_weights`` (quirks preserved).

    Every ``nn.Linear`` weight is redrawn with ``xavier_uniform_(gain=1)``.
    Biases are left at PyTorch's default uniform init — the official code
    calls ``xavier_uniform_`` on ``m.weight`` TWICE (the ``if m.bias`` branch
    repeats the weight init instead of touching the bias), so this function
    does exactly the same: weights only.
    """
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            init.xavier_uniform_(module.weight, gain=1.0)


class DeformationDecoder(nn.Module):
    """Shared trunk + ``dX/ds/dr`` heads (paper/default path)."""

    def __init__(self, feature_dim: int, config: DecoderConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = DecoderConfig()
        self.config = config
        layers: list[nn.Module] = [nn.Linear(feature_dim, config.width)]
        for _ in range(config.depth - 1):
            layers += [nn.ReLU(), nn.Linear(config.width, config.width)]
        self.trunk = nn.Sequential(*layers)
        self.pos_head = _head(config.width, 3)
        self.scale_head = _head(config.width, 3)
        self.rot_head = _head(config.width, 4)
        self.apply(_init_linear_weights)

    def forward(self, feature: torch.Tensor) -> DeformationDeltas:
        """Decode voxel features into raw-space deformation deltas.

        Args:
            feature: ``[N, F]`` HexPlane features (``F`` = constructor dim).

        Returns:
            :class:`DeformationDeltas` with ``[N, 3]`` / ``[N, 3]`` / ``[N, 4]``.
        """
        assert feature.dim() == 2, f"expected [N, F], got {tuple(feature.shape)}"
        hidden = self.trunk(feature)
        return DeformationDeltas(
            d_xyz=self.pos_head(hidden),
            d_scaling=self.scale_head(hidden),
            d_rotation=self.rot_head(hidden),
        )


class AuxDeformationHeads(nn.Module):
    """Official opacity/SH heads (NON-default path, disabled in practice).

    ``opacity`` predicts a raw-logit delta (``[N, 1]``); ``shs`` predicts
    ``[N, 48]`` reshaped to ``[N, 16, 3]`` raw-SH deltas. Kept for parity
    with the official module; the Commit 13+ field never calls this unless
    explicitly enabled.
    """

    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        width = (config or DecoderConfig()).width
        self.opacity_head = _head(width, 1)
        self.shs_head = _head(width, SH_COEFFS_PER_CHANNEL * 3)
        self.apply(_init_linear_weights)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: ``[N, W]`` trunk output. Returns ``([N, 1], [N, 16, 3])``."""
        return (
            self.opacity_head(hidden),
            self.shs_head(hidden).reshape(hidden.shape[0], SH_COEFFS_PER_CHANNEL, 3),
        )

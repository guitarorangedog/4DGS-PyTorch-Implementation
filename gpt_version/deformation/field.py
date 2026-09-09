"""Time-conditioned deformation field: the full 4D-GS motion model [4D].

Couples the Commit 11 HexPlane encoder with the Commit 12 decoder::

    (xyz, scaling_raw, rotation_raw, time)
      -> HexPlaneField(xyz, time) -> feature -> DeformationDecoder
      -> (dX, ds, dr) -> deformed canonical state

Faithful to official ``scene/deformation.py`` ``Deformation.forward_dynamic``
(default path). The official method takes positionally-encoded embeddings
but slices only their raw prefixes (``scales_emb[:, :3]`` is the raw input:
``poc_fre`` prepends the input), so operating directly on raw canonical
parameters is exactly equivalent and far more legible.

Exact default equations (``mask = 1``; official ``static_mlp``/``empty_voxel``
branches are dead config, not reproduced):

- ``X' = X + dX`` with ``dX`` unbounded, world units.
- ``s' = s_raw + ds`` in RAW LOG-scale space (``exp`` stays in the renderer).
- ``r' = r_raw + dr`` additive on the raw quaternion (official
  ``apply_rotation=False`` default; normalization stays in the model).
- opacity / SH pass through unchanged by default (official ``no_do`` /
  ``no_dshs``); see :class:`FieldConfig.enable_aux`.

Representation contract for Commit 14: returned ``scaling`` is RAW (feed
``model._scaling``-style tensors and pass the result on as raw), returned
``rotation`` is RAW (normalize downstream), ``xyz`` is world space.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from deformation.decoder import (
    AuxDeformationHeads,
    DecoderConfig,
    DeformationDecoder,
    DeformationDeltas,
)
from deformation.hexplane import HexPlaneConfig, HexPlaneField

__all__ = [
    "FieldConfig",
    "DeformedState",
    "DeformationField",
    "quaternion_multiply",
]


@dataclass
class FieldConfig:
    """Field options (official defaults)."""

    hexplane: HexPlaneConfig = field(default_factory=HexPlaneConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    apply_rotation: bool = False  #: official default False (additive quats).
    enable_aux: bool = False  #: opacity/SH heads; official default disabled.


@dataclass
class DeformedState:
    """Deformed canonical state (raw representations, Commit 14-ready)."""

    xyz: torch.Tensor  #: ``[N, 3]`` world positions.
    scaling: torch.Tensor  #: ``[N, 3]`` RAW log-scales.
    rotation: torch.Tensor  #: ``[N, 4]`` RAW quaternions.
    opacity: torch.Tensor | None = None  #: ``[N, 1]`` raw logits (if given).
    shs: torch.Tensor | None = None  #: ``[N, K, 3]`` raw SH (if given).


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product, normalized (official ``batch_quaternion_multiply``).

    Args:
        q1, q2: ``[N, 4]`` quaternions ``(w, x, y, z)``.

    Returns:
        ``[N, 4]`` unit quaternions ``q1 * q2``.
    """
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    q3 = torch.stack((w, x, y, z), dim=1)
    return q3 / q3.norm(dim=1, keepdim=True).clamp_min(1e-12)


class DeformationField(nn.Module):
    """``(canonical state, time) -> deformed state`` (paper Eq. 9)."""

    def __init__(self, config: FieldConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = FieldConfig()
        self.config = config
        self.hexplane = HexPlaneField(config.hexplane)
        self.decoder = DeformationDecoder(self.hexplane.feat_dim, config.decoder)
        self.aux = AuxDeformationHeads(config.decoder) if config.enable_aux else None

    # -- infrastructure access (Commits 15/16) --------------------------------
    @property
    def get_aabb(self):
        """HexPlane AABB ``(aabb[0], aabb[1])``."""
        return self.hexplane.get_aabb

    def set_aabb(self, xyz_max: torch.Tensor, xyz_min: torch.Tensor) -> None:
        """Install scene point-cloud bounds on the HexPlane."""
        self.hexplane.set_aabb(xyz_max, xyz_min)

    def get_grid_parameters(self) -> list[nn.Parameter]:
        """HexPlane plane parameters + AABB (grid optimizer group + TV losses).

        The AABB is included to mirror the official name-based grouping
        (``"grid" in name`` catches the official ``grid.aabb``); it has
        ``requires_grad=False`` so optimizers ignore it.
        """
        return [self.hexplane.aabb] + self.hexplane.get_grid_parameters()

    def get_mlp_parameters(self) -> list[nn.Parameter]:
        """Decoder (+ aux) MLP parameters (deformation optimizer group)."""
        params = list(self.decoder.parameters())
        if self.aux is not None:
            params += list(self.aux.parameters())
        return params

    # -- forward ----------------------------------------------------------------
    def forward(self, xyz: torch.Tensor, scaling: torch.Tensor,
                rotation: torch.Tensor, time,
                opacity: torch.Tensor | None = None,
                shs: torch.Tensor | None = None) -> DeformedState:
        """Deform canonical Gaussians to timestamp ``time``.

        Args:
            xyz: ``[N, 3]`` canonical positions (world).
            scaling: ``[N, 3]`` RAW log-scales (``model._scaling`` space).
            rotation: ``[N, 4]`` RAW quaternions (``model._rotation`` space).
            time: ``[N, 1]`` / ``[N]`` / scalar normalized timestamps.
            opacity: optional ``[N, 1]`` raw logits (passed through unless
                aux heads are enabled).
            shs: optional ``[N, K, 3]`` raw SH (ditto; aux requires ``K=16``).

        Returns:
            :class:`DeformedState` with RAW scaling/rotation (activations
            stay downstream, exactly as in the official pipeline).
        """
        n = xyz.shape[0]
        assert xyz.shape == (n, 3) and scaling.shape == (n, 3) and rotation.shape == (n, 4)
        feature = self.hexplane(xyz, time)

        if self.aux is not None:
            hidden = self.decoder.trunk(feature)
            deltas = DeformationDeltas(
                d_xyz=self.decoder.pos_head(hidden),
                d_scaling=self.decoder.scale_head(hidden),
                d_rotation=self.decoder.rot_head(hidden),
            )
            do, dshs = self.aux(hidden)
            assert opacity is not None and shs is not None and shs.shape[1] == 16
            opacity_out = opacity + do
            shs_out = shs + dshs
        else:
            deltas = self.decoder(feature)
            opacity_out, shs_out = opacity, shs

        if self.config.apply_rotation:
            rotation_out = quaternion_multiply(rotation, deltas.d_rotation)
        else:
            rotation_out = rotation + deltas.d_rotation
        return DeformedState(
            xyz=xyz + deltas.d_xyz,
            scaling=scaling + deltas.d_scaling,
            rotation=rotation_out,
            opacity=opacity_out,
            shs=shs_out,
        )

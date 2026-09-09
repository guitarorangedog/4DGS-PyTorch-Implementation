"""Canonical static 3D Gaussian representation [STATIC].

Paper Sec. 3.1: each Gaussian stores position ``X in R^3``, scale ``s``,
rotation quaternion ``r``, opacity ``alpha`` and SH color ``C``. This module
is the *canonical* set ``S`` that 4D-GS later deforms per timestamp
(``S'(t) = F(S, t)``, Eqs. 9-12); no time dependence lives here.

Raw stored parameters vs. activated properties (all shapes explicit):

- ``_xyz: [N, 3]`` raw == activated (identity). Property ``get_xyz``.
- ``_scaling: [N, 3]`` raw log-scales; ``get_scaling = exp(_scaling)``.
- ``_rotation: [N, 4]`` raw ``(w, x, y, z)`` quaternions (need not be unit);
  ``get_rotation = normalize(_rotation)``.
- ``_opacity: [N, 1]`` raw logits; ``get_opacity = sigmoid(_opacity)``.
- ``_features_dc: [N, 1, 3]`` raw SH DC coefficients (identity).
- ``_features_rest: [N, K-1, 3]`` raw higher-order SH (identity),
  with ``K = (max_sh_degree + 1) ** 2``.
- ``get_features -> [N, K, 3]`` concatenates DC and rest on dim 1.

Conventions match official 3D-GS / 4DGaussians ``scene/gaussian_model.py``:
same activations, same ``[N, 1, 3]`` / ``[N, K-1, 3]`` SH layout, same PLY
attribute order, same init constants (opacity ``0.1`` in activated space,
identity quaternions, ``RGB2SH`` DC seeding, ``log(sqrt(min_dist2))``
isotropic scales).

Documented deviations (no silent changes):

- Device-agnostic: official code hardcodes ``.cuda()`` everywhere; every
  method here takes a ``device`` argument (default CPU) so math tests run
  without a GPU. Formulas are unchanged.
- Scale init reference: official ``create_from_pcd`` calls
  ``simple_knn.distCUDA2`` (CUDA). :func:`nearest_sq_distances` below is a
  pure-PyTorch ``O(N^2)`` reference computing the same quantity (squared
  distance to the nearest *other* point, diagonal excluded, clamped at
  ``1e-7``). It is used for initialization/tests only and must not be
  confused with the CUDA rasterizer path (Commit 6). Production use at
  large ``N`` should swap in the CUDA kernel.
- No ``BasicPointCloud`` dependency: ``create_from_pointcloud`` takes plain
  ``[N, 3]`` tensors instead of the NumPy struct; the math is identical.
- No optimizer groups or densification here (Commits 5+); this class only
  holds parameters and (de)serializes them.
- Optimizer / densification buffers (``optimizer``, ``xyz_gradient_accum``,
  ``denom``, ``max_radii2D``) are declared below as ``None`` and owned by
  ``gaussians/densify.py`` (``attach_optimizer`` / ``init_densify_stats``);
  the full per-group training optimizer and schedulers arrive in Commit 8.
"""

import os

import numpy as np
import torch
from torch import nn

from gaussians.geometry import build_covariance, normalize_quaternion
from gaussians.sh import num_sh_coeffs, rgb_to_sh

__all__ = ["CanonicalGaussianModel", "nearest_sq_distances", "inverse_sigmoid"]

MIN_DIST2 = 1e-7
INIT_OPACITY = 0.1


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Logit function, inverse of ``sigmoid``.

    Args:
        x: ``[...]`` probabilities in ``(0, 1)``.

    Returns:
        ``[...]`` logits ``log(x / (1 - x))``.
    """
    return torch.log(x / (1 - x))


def nearest_sq_distances(points: torch.Tensor) -> torch.Tensor:
    """Squared distance from each point to its nearest *other* point.

    CPU/PyTorch reference for ``simple_knn.distCUDA2`` (see module docstring).
    ``O(N^2)`` memory — fine for init/tests, not for dense clouds.

    Args:
        points: ``[N, 3]`` positions, ``N >= 1``.

    Returns:
        ``[N]`` values ``min_{j != i} ||x_i - x_j||^2`` clamped at ``1e-7``.
        For the degenerate ``N == 1`` case there is no neighbor, so ``1.0``
        is returned (unit scale after ``log(sqrt(.))``).
    """
    assert points.dim() == 2 and points.shape[-1] == 3, (
        f"expected [N, 3], got {tuple(points.shape)}"
    )
    n = points.shape[0]
    if n == 1:
        return torch.ones((1,), dtype=points.dtype, device=points.device)
    d2 = torch.cdist(points, points, p=2).pow(2)
    d2.fill_diagonal_(float("inf"))
    return d2.min(dim=1).values.clamp_min(MIN_DIST2)


class CanonicalGaussianModel(nn.Module):
    """Learnable canonical static Gaussians (no time dependence)."""

    def __init__(self, max_sh_degree: int = 3) -> None:
        super().__init__()
        assert 0 <= max_sh_degree <= 3
        self.max_sh_degree = max_sh_degree
        self.active_sh_degree = 0
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, 0, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))
        # Owned by gaussians/densify.py; None until attach_optimizer /
        # init_densify_stats run (Commit 5). Declared here so attribute
        # access is explicit rather than dynamic.
        self.optimizer: torch.optim.Optimizer | None = None
        self.percent_dense: float = 0.01
        #: Scene-extent multiplier for the position LR (official
        #: ``spatial_lr_scale``, set from the scene radius). The full static
        #: optimizer + xyz schedule is built by
        #: ``training/optim.setup_static_optimizer`` (Commit 8).
        self.spatial_lr_scale: float = 1.0
        self.xyz_gradient_accum: torch.Tensor | None = None
        self.denom: torch.Tensor | None = None
        self.max_radii2D: torch.Tensor | None = None
        #: Exponential xyz LR schedule, installed by ``setup_static_optimizer``.
        self.xyz_schedule = None
        #: Deformation/grid schedules, installed by ``setup_4d_optimizer`` (Commit 16).
        self.deform_schedule = None
        self.grid_schedule = None

    def static_param_groups(self) -> list[tuple[str, nn.Parameter]]:
        """Named ``(group_name, parameter)`` pairs for optimizer construction.

        Order and names match the official ``training_setup`` static subset
        (``xyz``, ``f_dc``, ``f_rest``, ``opacity``, ``scaling``, ``rotation``)
        so Commit 5 surgery and the Commit 8 trainer share one contract.
        """
        return [
            ("xyz", self._xyz),
            ("f_dc", self._features_dc),
            ("f_rest", self._features_rest),
            ("opacity", self._opacity),
            ("scaling", self._scaling),
            ("rotation", self._rotation),
        ]

    # -- activated accessors -------------------------------------------------
    @property
    def get_xyz(self) -> torch.Tensor:
        """``[N, 3]`` positions (raw == activated)."""
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        """``[N, 3]`` scales in world units: ``exp(_scaling)``."""
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        """``[N, 4]`` unit quaternions: ``normalize(_rotation)``."""
        return normalize_quaternion(self._rotation)

    @property
    def get_opacity(self) -> torch.Tensor:
        """``[N, 1]`` opacities: ``sigmoid(_opacity)``."""
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        """``[N, K, 3]`` SH coefficients (DC ++ rest)."""
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """``[N, 3, 3]`` covariances ``Sigma = L L^T`` (paper Eq. 2)."""
        return build_covariance(self.get_scaling * scaling_modifier, self.get_rotation)

    @property
    def num_points(self) -> int:
        """Number of Gaussians ``N``."""
        return self._xyz.shape[0]

    def oneupSHdegree(self) -> None:
        """Increase the active SH degree by one (capped at max)."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # -- initialization ------------------------------------------------------
    @torch.no_grad()
    def create_from_pointcloud(
        self, points: torch.Tensor, colors: torch.Tensor, device: torch.device | str = "cpu"
    ) -> None:
        """Initialize canonical parameters from a colored point cloud.

        Mirrors official ``create_from_pcd`` (same constants and layout):

        - ``_xyz`` = input points.
        - ``_features_dc`` = ``RGB2SH(colors)`` at band 0, rest zeros.
        - ``_scaling`` = ``log(sqrt(nearest_dist2))`` isotropic (all 3 axes).
        - ``_rotation`` = ``[1, 0, 0, 0]`` identity.
        - ``_opacity`` = ``inverse_sigmoid(0.1)``.

        Args:
            points: ``[N, 3]`` positions.
            colors: ``[N, 3]`` linear RGB in ``[0, 1]``.
            device: target device for the parameters.
        """
        assert points.shape == colors.shape and points.dim() == 2 and points.shape[-1] == 3
        device = torch.device(device)
        n = points.shape[0]
        k = num_sh_coeffs(self.max_sh_degree)
        pts = points.to(dtype=torch.float32, device=device)
        col = colors.to(dtype=torch.float32, device=device).clamp(0.0, 1.0)

        fused_sh = torch.zeros((n, 3, k), dtype=torch.float32, device=device)
        fused_sh[:, :, 0] = rgb_to_sh(col)

        dist2 = nearest_sq_distances(pts)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((n, 4), dtype=torch.float32, device=device)
        rots[:, 0] = 1.0
        opacities = inverse_sigmoid(
            INIT_OPACITY * torch.ones((n, 1), dtype=torch.float32, device=device)
        )

        self._xyz = nn.Parameter(pts.requires_grad_(True))
        self._features_dc = nn.Parameter(
            fused_sh[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            fused_sh[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.active_sh_degree = 0

    # -- PLY serialization ---------------------------------------------------
    def construct_list_of_attributes(self) -> list[str]:
        """PLY attribute order (identical to the official implementation)."""
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        attrs += ["f_dc_%d" % i for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])]
        attrs += ["f_rest_%d" % i for i in range(self._features_rest.shape[1] * self._features_rest.shape[2])]
        attrs += ["opacity"]
        attrs += ["scale_%d" % i for i in range(self._scaling.shape[1])]
        attrs += ["rot_%d" % i for i in range(self._rotation.shape[1])]
        return attrs

    def save_ply(self, path: str) -> None:
        """Write parameters to PLY.

        Convention (matches official ``save_ply``): the file stores **raw
        internal values** — log-scales, unnormalized quaternions, opacity
        logits, raw SH — so loading reconstructs the model bit-for-bit.
        Positions are activated (= raw). Normals are zeros (placeholder).
        """
        from plyfile import PlyData, PlyElement

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self.construct_list_of_attributes()]
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_ply(self, path: str, device: torch.device | str = "cpu") -> "CanonicalGaussianModel":
        """Load parameters from PLY (inverse of :meth:`save_ply`).

        Reads **raw** values (see :meth:`save_ply`) and sets
        ``active_sh_degree = max_sh_degree``, mirroring official ``load_ply``.

        Args:
            path: PLY file written by :meth:`save_ply`.
            device: target device for the parameters.

        Returns:
            ``self`` for chaining.
        """
        from plyfile import PlyData

        device = torch.device(device)
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_names = sorted(extra_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_names)))
        for idx, name in enumerate(extra_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
        return self

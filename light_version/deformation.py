"""HexPlane-lite deformation field: (X, t) -> RAW offsets (dmean, dscale, dquat).

Idea: factorize 4D space-time into six learned 2D planes (xy,xz,xt,yz,yt,zt).
Bilinear-sample all six, fuse by product, decode with a tiny MLP.
Outputs are RAW offsets added BEFORE activation elsewhere:
  mean_t = mean + dmean, log_scale_t = log_scale + dscale, quat_raw_t = quat_raw + dquat.
Final decoder layer is zero-initialized, so the field starts as identity (delta ~ 0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PLANES = ["xy", "xz", "xt", "yz", "yt", "zt"]


class HexPlaneLite(nn.Module):
    """One resolution (R=32, D=8): six [1,D,R,R] planes -> fused feature [N,D]."""

    def __init__(self, res=32, dim=8):
        super().__init__()
        self.res, self.dim = res, dim
        for name in PLANES:  # random init on ALL six (0.5 scale keeps the
            # 6-way product visible); decoder starts at zero anyway, so delta ~ 0
            self.register_parameter(name, nn.Parameter(torch.randn(1, dim, res, res) * 0.5))

    def sample(self, plane, a, b):
        """Sample plane P(a,b) at coords a,b in [-1,1] -> [N,D].

        Storage: dim2 (H) <-> first coord a, dim3 (W) <-> second coord b.
        grid_sample wants (x=width=b, y=height=a), so grid = stack([b, a])."""
        N = a.shape[0]
        grid = torch.stack([b, a], dim=-1).view(1, 1, N, 2)
        return F.grid_sample(plane, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)[0, :, 0, :].T

    def forward(self, pos, t):
        """pos [N,3] (clamped to [-1,1]), t float or tensor with 1 or N values."""
        x, y, z = pos[:, 0].clamp(-1, 1), pos[:, 1].clamp(-1, 1), pos[:, 2].clamp(-1, 1)
        tg = torch.as_tensor(t, dtype=pos.dtype, device=pos.device).reshape(-1)
        tg = tg.expand(pos.shape[0]) if tg.numel() == 1 else tg.reshape(-1)
        tg = (2 * tg - 1).clamp(-1, 1)  # time [0,1] -> grid coords [-1,1]
        f = {"xy": self.sample(self.xy, x, y), "xz": self.sample(self.xz, x, z),
             "xt": self.sample(self.xt, x, tg), "yz": self.sample(self.yz, y, z),
             "yt": self.sample(self.yt, y, tg), "zt": self.sample(self.zt, z, tg)}
        # Product fusion (paper spirit): space part x time part; each part is
        # itself a product, so the feature is high only where ALL planes agree.
        return (f["xy"] * f["xz"] * f["yz"]) * (f["xt"] * f["yt"] * f["zt"])


class DeformationField(nn.Module):
    """HexPlaneLite + tiny MLP: feature [N,D] -> (dmean [N,3], dscale [N,3], dquat [N,4])."""

    def __init__(self, dim=8):
        super().__init__()
        self.hexplane = HexPlaneLite(dim=dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 10))
        nn.init.zeros_(self.decoder[2].weight)  # identity init: deformation ~ 0
        nn.init.zeros_(self.decoder[2].bias)

    def forward(self, pos, t):
        d = self.decoder(self.hexplane(pos, t))  # [N,10]
        return d[:, 0:3], d[:, 3:6], d[:, 6:10]


if __name__ == "__main__":
    pos = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0],
                        [0.0, 0.7, 0.0], [0.0, 0.0, 0.5]])
    field = DeformationField()
    for t in [0.0, 0.5, 1.0]:
        dm, ds, dq = field(pos, t)
        ok = all(torch.isfinite(v).all() for v in (dm, ds, dq))
        print(f"t={t}: shapes={tuple(dm.shape)},{tuple(ds.shape)},{tuple(dq.shape)} "
              f"|dmean|={dm.abs().mean():.6f} |dscale|={ds.abs().mean():.6f} "
              f"|dquat|={dq.abs().mean():.6f} finite={ok}")
    f0, f1 = field.hexplane(pos, 0.0), field.hexplane(pos, 1.0)
    print(f"time-conditioned: |feat(t=0)-feat(t=1)| mean={(f0 - f1).abs().mean():.6f} (> 0 = time matters)")

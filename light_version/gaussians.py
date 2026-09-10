"""N canonical 3D Gaussians: RAW optimized params -> ACTIVATED physical params.

Raw (what Adam updates, unconstrained):
  means          [N,3]  direct
  log_scales     [N,3]  log-space, so scales stay positive
  quats_raw      [N,4]  unnormalized quaternion [w,x,y,z]
  opacity_logits [N,1]  logit-space, so opacity stays in (0,1)
  colors_raw     [N,3]  unbounded, so colors stay in (0,1)

Activated (what the renderer consumes):
  means                     identity
  scales      = exp(log_scales)            > 0
  quats       = quats_raw / ||quats_raw||  unit norm
  opacity     = sigmoid(opacity_logits)    in (0,1)
  colors      = sigmoid(colors_raw)        in (0,1)

No covariance / rotation matrices / SH / deformation here (later commits).
"""

import math

import torch
import torch.nn as nn


class CanonicalGaussians(nn.Module):
    def __init__(self, means, log_scales, quats_raw, opacity_logits, colors_raw):
        super().__init__()
        self.means = nn.Parameter(means)                    # [N,3] direct
        self.log_scales = nn.Parameter(log_scales)          # [N,3] -> exp
        self.quats_raw = nn.Parameter(quats_raw)            # [N,4] [w,x,y,z] -> normalize
        self.opacity_logits = nn.Parameter(opacity_logits)  # [N,1] -> sigmoid
        self.colors_raw = nn.Parameter(colors_raw)          # [N,3] -> sigmoid

    @classmethod
    def random(cls, N=8, extent=1.0, seed=0):
        """Tiny init near the origin; every line is the whole strategy."""
        g = torch.Generator().manual_seed(seed)
        means = (torch.rand((N, 3), generator=g) * 2 - 1) * extent
        log_scales = math.log(0.2) + 0.2 * torch.randn((N, 3), generator=g)
        quats_raw = torch.tensor([1.0, 0.0, 0.0, 0.0]) + 0.1 * torch.randn((N, 4), generator=g)
        opacity_logits = torch.zeros((N, 1))  # sigmoid(0) = 0.5
        colors_raw = torch.randn((N, 3), generator=g)  # sigmoid -> diverse RGB
        return cls(means, log_scales, quats_raw, opacity_logits, colors_raw)

    def activated(self):
        """Return physical params the renderer will use."""
        scales = torch.exp(self.log_scales)
        quats = self.quats_raw / self.quats_raw.norm(dim=1, keepdim=True)
        opacity = torch.sigmoid(self.opacity_logits)
        colors = torch.sigmoid(self.colors_raw)
        return self.means, scales, quats, opacity, colors


if __name__ == "__main__":
    gs = CanonicalGaussians.random(N=5, extent=0.8, seed=0)
    print(f"raw shapes: means={tuple(gs.means.shape)} log_scales={tuple(gs.log_scales.shape)} "
          f"quats={tuple(gs.quats_raw.shape)} opacity={tuple(gs.opacity_logits.shape)} "
          f"colors={tuple(gs.colors_raw.shape)}")
    means, scales, quats, opacity, colors = gs.activated()
    print(f"scales  > 0: min={scales.min():.4f} max={scales.max():.4f}")
    print(f"quat norms ~ 1: {quats.norm(dim=1).tolist()}")
    print(f"opacity in (0,1): min={opacity.min():.4f} max={opacity.max():.4f}")
    print(f"colors  in (0,1): min={colors.min():.4f} max={colors.max():.4f}")

"""Tiny synthetic dynamic scene: 3 blobs with analytic motion, t in [0,1].

Concept: means(t) = base + motion(t). Only positions move; scales,
rotations, opacity, colors are fixed. Later, the learned deformation
field will approximate this same mapping: canonical + field(X, t).
32 observations = 4 orbit cameras x 8 timestamps, precomputed with the
same renderer under no_grad (fixed targets, not learned).
"""

import math

import torch
from torch.utils.data import Dataset

from cameras import PinholeCamera, look_at
from render import render

TIMES = torch.linspace(0, 1, 8)          # 8 timestamps in [0, 1]
COLORS = torch.tensor([[0.9, 0.1, 0.1],  # red:   moves horizontally
                       [0.1, 0.9, 0.1],  # green: moves vertically
                       [0.1, 0.1, 0.9]]) # blue:  small circle
SCALES = torch.full((3, 3), 0.25)
QUATS = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
OPACITY = torch.full((3, 1), 0.9)


def ground_truth_frame(camera, t):
    """Analytic ground truth at ARBITRARY continuous t (reuses means_at)."""
    with torch.no_grad():
        return render(camera, means_at(t), SCALES, QUATS, OPACITY, COLORS)


def means_at(t):
    """Analytic trajectories (t is a float in [0, 1]); each is one line."""
    t = float(t)
    red = torch.tensor([-0.6 + 1.2 * t, 0.0, 0.0])                       # left -> right
    green = torch.tensor([0.6, -0.5 + 1.0 * t, 0.0])                     # bottom -> top
    blue = torch.tensor([0.3 * math.cos(2 * math.pi * t),                # circle
                         0.2 + 0.3 * math.sin(2 * math.pi * t), 0.0])
    return torch.stack([red, green, blue])


def make_cameras():
    """4 cameras on a horizontal orbit (radius 3, height 0.5), looking at origin."""
    cams = []
    for deg in [0, 90, 180, 270]:
        a = math.radians(deg)
        R, t = look_at(eye=(3 * math.sin(a), 0.5, 3 * math.cos(a)),
                       center=(0.0, 0.0, 0.0))
        cams.append(PinholeCamera(R, t, fx=64.0, fy=64.0, cx=32.0, cy=32.0, H=64, W=64))
    return cams


class TinyDynamicDataset(Dataset):
    """Precomputed fixed targets. Get one sample at a time (no batched
    DataLoader: samples carry a PinholeCamera object, and training will
    sample random (camera, time) pairs per iteration)."""

    def __init__(self):
        self.cameras = make_cameras()
        self.times = TIMES
        with torch.no_grad():  # fixed ground truth, never optimized
            self.images = {(c, k): render(self.cameras[c], means_at(t),
                                          SCALES, QUATS, OPACITY, COLORS)
                           for c in range(len(self.cameras)) for k, t in enumerate(self.times)}

    def __len__(self):
        return len(self.cameras) * len(self.times)  # 4 x 8 = 32

    def __getitem__(self, i):
        c, k = divmod(i, len(self.times))
        return {"image": self.images[(c, k)], "time": self.times[k],
                "camera": self.cameras[c], "cam_index": c}


if __name__ == "__main__":
    ds = TinyDynamicDataset()
    print(f"samples={len(ds)} cameras={len(ds.cameras)} times={ds.times.tolist()}")
    for t in [0.0, 0.5, 1.0]:
        print(f"t={t}: means={means_at(t).tolist()}")
    s = ds[0]
    print(f"sample: image={tuple(s['image'].shape)} time={s['time'].item()} "
          f"finite={torch.isfinite(s['image']).all().item()} "
          f"min={s['image'].min():.3f} max={s['image'].max():.3f}")
    f0, f1, f2 = ds.images[(0, 0)], ds.images[(0, 3)], ds.images[(0, 7)]
    print(f"frame change cam0: |t0-t.43|={(f0 - f1).abs().mean():.4f} "
          f"|t0-t1|={(f0 - f2).abs().mean():.4f} (both should be clearly > 0)")
    try:
        from PIL import Image
        import numpy as np
        for name, img in [("t0", f0), ("t05", ds.images[(0, 4)]), ("t1", f2)]:
            Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)
                            ).save(f"/tmp/light_dynamic_{name}.png")
        print("saved /tmp/light_dynamic_t0.png t05 t1 (camera 0)")
    except ImportError:
        pass

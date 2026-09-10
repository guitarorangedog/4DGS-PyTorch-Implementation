"""Commit 10: novel-TIME demo. Train once, then render unseen timestamps.

t is CONTINUOUS: training saw only 8 timestamps, but field(canonical, t)
evaluates at ANY t in [0,1] with NO further optimization. Predictions are
checked against analytic ground truth (means_at works at any t); outputs go
to light_version/out/ (strip + prediction/GT comparison + GIF).
"""

import math
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image

from dataset import ground_truth_frame
from render import render_dynamic
from train import train_model

NOVEL_TIMES = [0.07, 0.21, 0.36, 0.50, 0.64, 0.79, 0.93]


def main():
    gs, field, ds = train_model()       # train ONCE; no optimization below
    cam = ds.cameras[0]                 # novel TIME demo (camera fixed)
    train_set = {round(t.item(), 4) for t in ds.times}
    print(f"training times: {sorted(train_set)}")
    print(f"novel times: {NOVEL_TIMES} "
          f"(all unseen: {all(round(t, 4) not in train_set for t in NOVEL_TIMES)})")
    with torch.no_grad():
        l1s, psnrs = [], []
        for t in NOVEL_TIMES:           # pure evaluation at unseen t
            pred = render_dynamic(cam, gs, field, t)
            gt = ground_truth_frame(cam, t)
            assert pred.shape == (64, 64, 3) and torch.isfinite(pred).all()
            assert 0.0 <= pred.min() and pred.max() <= 1.0
            l1 = (pred - gt).abs().mean().item()
            psnr = -10 * math.log10(((pred - gt) ** 2).mean().item())
            l1s.append(l1)
            psnrs.append(psnr)
            print(f"t={t:.2f} | L1={l1:.5f} | PSNR={psnr:.2f}")
        print(f"novel-time mean: L1={sum(l1s) / len(l1s):.5f} "
              f"PSNR={sum(psnrs) / len(psnrs):.2f}")
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:  # learned continuous trajectory
            print(f"t={t:.2f}: {gs.deformed_activated(field, t)[0].tolist()}")
        out = Path(__file__).parent / "out"
        out.mkdir(exist_ok=True)
        dense = torch.linspace(0, 1, 21).tolist()
        preds = [render_dynamic(cam, gs, field, t) for t in dense]
        gts = [ground_truth_frame(cam, t) for t in dense]
        smooth = sum((a - b).abs().mean().item()
                     for a, b in zip(preds[1:], preds[:-1])) / (len(preds) - 1)
        print(f"mean consecutive-frame change = {smooth:.4f} (>0 = smooth motion)")
        to8 = lambda im: (im.clamp(0, 1).numpy() * 255).astype(np.uint8)
        Image.fromarray(np.concatenate([to8(im) for im in preds], axis=1)
                        ).save(out / "interpolation_strip.png")
        Image.fromarray(np.concatenate(
            [np.concatenate([to8(im) for im in preds], axis=1),
             np.concatenate([to8(im) for im in gts], axis=1)], axis=0)
            ).save(out / "interpolation_compare.png")
        imageio.mimsave(out / "interpolation.gif", [to8(im) for im in preds], fps=7)
        print(f"saved {out}/interpolation_strip.png compare.png interpolation.gif")


if __name__ == "__main__":
    main()

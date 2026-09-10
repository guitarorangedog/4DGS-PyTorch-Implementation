"""Commit 9: minimal coarse-to-fine 4DGS training on the tiny dynamic scene.

coarse = static warmup: fit canonical Gaussians to all 4 cameras at ONE
  existing reference timestamp with the STATIC renderer (field frozen).
fine = joint dynamic fit: optimize canonical Gaussians + deformation field
  on random (camera, time) samples with render_dynamic(). Plain L1 + Adam.
(Simplified educational reading of the paper's coarse-to-fine schedule.)
"""

import math

import torch

from dataset import TinyDynamicDataset
from deformation import DeformationField
from gaussians import CanonicalGaussians
from render import render, render_dynamic

torch.set_num_threads(min(8, torch.get_num_threads()))  # tiny tensors: avoid many-thread overhead


def evaluate(ds, gs, field):
    """Full 32-observation mean L1 + PSNR (reporting only, no gradients)."""
    l1, mse = 0.0, 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            d = render_dynamic(s["camera"], gs, field, s["time"]) - s["image"]
            l1 += d.abs().mean().item()
            mse += (d ** 2).mean().item()
    return l1 / len(ds), -10 * math.log10(mse / len(ds))


def train_model(coarse_iters=300, fine_iters=1500, coarse_lr=0.05, fine_lr=0.005, seed=0):
    """Full coarse-to-fine fit; returns (canonical Gaussians, field, dataset)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    ds = TinyDynamicDataset()                       # 32 fixed targets: 4 cams x 8 times
    ref = len(ds.times) // 2                        # middle existing observation
    t_ref = ds.times[ref]
    print(f"reference timestamp: index {ref} -> t={t_ref.item():.4f} (existing, not invented)")
    gs = CanonicalGaussians.random(N=3, extent=0.8, seed=seed)
    field = DeformationField()                      # zero final layer: identity at start

    # STAGE A (coarse): static fit at t_ref, field frozen (not in optimizer).
    opt = torch.optim.Adam(gs.parameters(), lr=coarse_lr)
    for i in range(coarse_iters + 1):
        c = torch.randint(0, len(ds.cameras), (1,), generator=g).item()
        s = ds[c * len(ds.times) + ref]             # random camera, fixed t_ref
        loss = (render(s["camera"], *gs.activated()) - s["image"]).abs().mean()
        if i % 100 == 0:
            print(f"[coarse] iter {i:4d} | L1 = {loss.item():.5f}")
        if i < coarse_iters:
            opt.zero_grad()
            loss.backward()
            opt.step()

    l1_before, psnr_before = evaluate(ds, gs, field)
    print(f"after coarse: full-dataset L1 = {l1_before:.5f} PSNR = {psnr_before:.2f}")

    # STAGE B (fine): joint fit on random (camera, time) samples.
    # One small shared rate: coarse already placed canonical well, so fine
    # only refines it while the field grows deltas from zero. (A large rate
    # here thrashes canonical across conflicting timestamps and ejects
    # Gaussians out of view, killing the field's gradients.)
    opt = torch.optim.Adam(list(gs.parameters()) + list(field.parameters()), lr=fine_lr)
    first_grads, mid_grads = None, None
    for i in range(fine_iters + 1):
        s = ds[torch.randint(0, len(ds), (1,), generator=g).item()]
        loss = (render_dynamic(s["camera"], gs, field, s["time"]) - s["image"]).abs().mean()
        if i % 300 == 0:
            print(f"[fine]   iter {i:4d} | L1 = {loss.item():.5f}")
        if i < fine_iters:
            opt.zero_grad()
            loss.backward()
            if i == 0:  # Commit-8 behavior live: final layer learns first, HexPlane ~0
                first_grads = (field.decoder[2].weight.grad.abs().sum().item(),
                               None if field.hexplane.xy.grad is None else
                               field.hexplane.xy.grad.abs().sum().item())
            if i == 200:  # after final weights move, HexPlane should receive grads
                mid_grads = field.hexplane.xy.grad.abs().sum().item()
            opt.step()
    print(f"first fine step: final-layer grad={first_grads[0]:.2e}, hexplane grad={first_grads[1]}")
    print(f"later fine step: hexplane grad={mid_grads:.2e} (>0 = field is learning)")

    l1_after, psnr_after = evaluate(ds, gs, field)
    print(f"after fine: full-dataset L1 = {l1_after:.5f} PSNR = {psnr_after:.2f} "
          f"| improved = {l1_after < l1_before}")

    with torch.no_grad():  # deformation really learned? positions must vary with time
        t_mid = ds.times[ref]
        m0 = gs.deformed_activated(field, 0.0)[0]
        mm = gs.deformed_activated(field, t_mid)[0]
        m1 = gs.deformed_activated(field, 1.0)[0]
        print(f"deformed means t=0:   {m0.tolist()}")
        print(f"deformed means t=mid: {mm.tolist()}")
        print(f"deformed means t=1:   {m1.tolist()}")
        print(f"mean |means(t=0)-means(t=1)| = {(m0 - m1).abs().mean().item():.4f} (>0 = dynamic)")
        f0 = render_dynamic(ds.cameras[0], gs, field, ds.times[0])
        f1 = render_dynamic(ds.cameras[0], gs, field, ds.times[-1])
        print(f"cam0 |frame(t=0)-frame(t=1)| = {(f0 - f1).abs().mean().item():.4f} (>0 = visibly dynamic)")
        try:
            from PIL import Image
            import numpy as np
            for name, t in [("t0", ds.times[0]), ("tmid", t_mid), ("t1", ds.times[-1])]:
                img = render_dynamic(ds.cameras[0], gs, field, t)
                Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)
                                ).save(f"/tmp/light_4dgs_{name}.png")
            print("saved /tmp/light_4dgs_t0.png tmid t1 (camera 0)")
        except ImportError:
            pass
    return gs, field, ds


if __name__ == "__main__":
    train_model()

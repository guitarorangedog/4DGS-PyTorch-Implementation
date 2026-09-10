"""Commit 5: fit STATIC Gaussians from one view (no time yet).

Loop taught here:
  raw params -> activated() -> render() -> L1 vs target -> backward -> Adam.
Target image is made with the SAME renderer, so this tests gradients,
not rendering fidelity. Fixed N=3, one 64x64 camera, plain Adam + L1.
"""

import torch

from cameras import PinholeCamera, look_at
from gaussians import CanonicalGaussians
from render import render

torch.set_num_threads(min(8, torch.get_num_threads()))  # tiny 64x64 tensors: avoid many-thread overhead


def make_camera():
    R, t = look_at(eye=(0.0, 0.0, 3.0), center=(0.0, 0.0, 0.0))
    return PinholeCamera(R, t, fx=64.0, fy=64.0, cx=32.0, cy=32.0, H=64, W=64)


def make_target():
    """3 hand-placed Gaussians: red-left, green-right, blue-top."""
    means = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.7, 0.0]])
    log_scales = torch.full((3, 3), -1.3863)          # log(0.25)
    quats_raw = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    opacity_logits = torch.full((3, 1), 2.1972)       # sigmoid = 0.9
    colors_raw = torch.tensor([[2.2, -2.2, -2.2],     # ~red
                               [-2.2, 2.2, -2.2],     # ~green
                               [-2.2, -2.2, 2.2]])    # ~blue
    return CanonicalGaussians(means, log_scales, quats_raw, opacity_logits, colors_raw)


def main(iters=300, lr=0.05, seed=0):
    torch.manual_seed(seed)
    cam = make_camera()
    with torch.no_grad():                             # target is NOT optimized
        target = render(cam, *make_target().activated())

    gs = CanonicalGaussians.random(N=3, extent=0.8, seed=seed)  # learnable
    opt = torch.optim.Adam(gs.parameters(), lr=lr)    # all raw params, one lr
    init_loss = None
    for i in range(iters + 1):
        image = render(cam, *gs.activated())
        loss = (image - target).abs().mean()          # plain L1
        if i == 0:
            init_loss = loss.item()
        if i % 50 == 0:
            print(f"iter {i:4d} | L1 = {loss.item():.5f}")
        if i < iters:
            opt.zero_grad()
            loss.backward()
            opt.step()
    final_loss = loss.item()
    means, _, _, _, colors = gs.activated()
    print(f"initial L1 = {init_loss:.5f} | final L1 = {final_loss:.5f} "
          f"| improved = {final_loss < init_loss}")
    print(f"learned means: {means.detach().tolist()}")
    print(f"learned colors: {colors.detach().tolist()}")
    try:
        from PIL import Image
        import numpy as np
        Image.fromarray((target.clamp(0, 1).numpy() * 255).astype(np.uint8)).save("/tmp/light_target.png")
        Image.fromarray((image.detach().clamp(0, 1).numpy() * 255).astype(np.uint8)).save("/tmp/light_fitted.png")
        print("saved /tmp/light_target.png /tmp/light_fitted.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

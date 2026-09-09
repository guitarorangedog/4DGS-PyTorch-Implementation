"""GPU smoke gate for gaussians/rasterizer.py (Commit 6).

- Without CUDA / without the extension: asserts :func:`render_view` fails
  fast with ``RuntimeError`` (no silent fallback). Runs anywhere.
- With both: forward + backward smoke on a tiny synthetic scene
  (real ``GaussianRasterizer`` on GPU).

Run via ``python3 -m tests.test_rasterizer_smoke``.
"""

import numpy as np
import torch

from data.cameras import Camera
from gaussians.gaussian_model import CanonicalGaussianModel
from gaussians.rasterizer import render_view


def _extension_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import diff_gaussian_rasterization  # noqa: F401
        return True
    except ImportError:
        return False


def _tiny_scene(device: torch.device):
    torch.manual_seed(0)
    # COLMAP/+z-forward convention: with R=I the camera at the origin looks
    # along +z, so points must have POSITIVE camera-space z to be visible.
    cam = Camera(R=np.eye(3), T=np.zeros(3),
                 FoVx=1.0, FoVy=1.0, image_width=64, image_height=64, time=0.0)
    n = 8
    pts = torch.tensor([
        [0.0, 0.0, 2.5], [0.6, 0.0, 2.8], [-0.6, 0.2, 2.2], [0.1, -0.5, 3.0],
        [0.3, 0.4, 1.8], [-0.2, -0.3, 2.6], [0.0, 0.6, 3.2], [0.5, -0.4, 2.0],
    ])
    col = torch.rand(n, 3) * 0.6 + 0.2
    model = CanonicalGaussianModel(max_sh_degree=0).to(device)
    model.create_from_pointcloud(pts, col, device=device)
    with torch.no_grad():
        model._scaling.copy_(torch.full((n, 3), float(np.log(0.15)), device=device))
    model.active_sh_degree = 0
    return cam, model


def test_missing_extension_fails_fast() -> None:
    if _extension_available():
        print("test_missing_extension_fails_fast: skipped (extension present)")
        return
    cam = Camera(R=np.eye(3), T=np.zeros(3), FoVx=1.0, FoVy=1.0,
                 image_width=16, image_height=16)
    model = CanonicalGaussianModel(max_sh_degree=0)
    model.create_from_pointcloud(torch.zeros(2, 3), torch.full((2, 3), 0.5))
    try:
        render_view(cam, model)
    except RuntimeError:
        print("test_missing_extension_fails_fast: passed (RuntimeError)")
        return
    raise AssertionError("render_view must raise RuntimeError without the extension")


def test_forward_backward_smoke() -> None:
    if not _extension_available():
        print("test_forward_backward_smoke: skipped (no CUDA/extension)")
        return
    device = torch.device("cuda")
    cam, model = _tiny_scene(device)

    pkg = render_view(cam, model, bg_color=(1.0, 1.0, 1.0), device=device)
    img = pkg["render"]
    assert img.shape == (3, 64, 64), img.shape
    assert torch.isfinite(img).all()
    assert (img < 0.999).any(), "all-background image: Gaussians missed the frame"
    assert pkg["visibility_filter"].any(), "no Gaussian visible"
    assert pkg["radii"].max() > 0
    if pkg["depth"] is not None:
        vis = pkg["visibility_filter"]
        assert torch.isfinite(pkg["depth"]).all()

    loss = img.mean()
    assert torch.isfinite(loss)
    loss.backward()

    assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all()
    assert model._xyz.grad.abs().sum() > 0, "no gradient reached xyz"
    assert pkg["viewspace_points"].grad is not None, "viewspace grads missing"
    assert torch.isfinite(pkg["viewspace_points"].grad).all()
    for name in ("_features_dc", "_scaling", "_opacity"):
        g = getattr(model, name).grad
        assert g is not None and torch.isfinite(g).all(), name
    print(f"test_forward_backward_smoke: passed "
          f"(mean={loss.item():.4f}, visible={int(pkg['visibility_filter'].sum())}, "
          f"|dL/dxyz|={model._xyz.grad.abs().sum().item():.4e})")


if __name__ == "__main__":
    test_missing_extension_fails_fast()
    test_forward_backward_smoke()
    print("test_rasterizer_smoke.py: done")

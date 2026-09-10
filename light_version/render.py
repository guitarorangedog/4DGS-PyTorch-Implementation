"""Pure-PyTorch static Gaussian splat renderer (educational, brute force).

Pipeline (takes ACTIVATED params from gaussians.py, so the raw->active
boundary stays obvious: caller must pass gaussians.activated() here):
  quat [w,x,y,z] -> R -> Sigma_3d = R diag(s^2) R^T
  Sigma_cam = Rc Sigma_world Rc^T,  J (perspective) -> Sigma_2d = J Sigma_cam J^T
  per-pixel 2D Gaussian * opacity -> depth-sorted front-to-back alpha blend
Background is white (1,1,1). O(N*H*W); fine for 64x64 and N ~ tens.
Everything stays in torch so Commit 5 can differentiate through it.
"""

import torch

from cameras import PinholeCamera, look_at


def quat_to_rotmat(q):
    """q [N,4] normalized [w,x,y,z] -> R [N,3,3]. Explicit formula, no library."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


def render(camera, means, scales, quats, opacity, colors, eps=1e-4):
    """All inputs ACTIVATED. means [N,3], scales [N,3], quats [N,4],
    opacity [N,1] or [N], colors [N,3] in [0,1]. Returns image [H,W,3]."""
    H, W = camera.H, camera.W
    device = means.device
    op = opacity.reshape(-1)
    cam = camera.world_to_cam(means)          # [N,3], translation drops out of cov
    depth = -cam[:, 2]                        # positive in front (-z convention)
    visible = depth > 0.2                     # ignore behind / too-close points
    if not visible.any():
        return torch.ones(H, W, 3, device=device)

    # 3D covariance in world, then rotated (not translated) into camera frame.
    R = quat_to_rotmat(quats)                                     # [N,3,3]
    Sigma_world = R @ torch.diag_embed(scales ** 2) @ R.transpose(1, 2)
    Rc = camera.R.to(device)
    Sigma_cam = Rc @ Sigma_world @ Rc.T                           # [N,3,3]

    # Perspective Jacobian for u = fx*x/d+cx, v = cy-fy*y/d, d = -z:
    #   du/dx = fx/d,  du/dz = +fx*x/d^2   (d/dz = -1 flips the usual sign)
    #   dv/dy = -fy/d, dv/dz = -fy*y/d^2   (extra minus: v rows grow down)
    x, y, d = cam[:, 0], cam[:, 1], depth
    J = torch.zeros(len(means), 2, 3, device=device)
    J[:, 0, 0] = camera.fx / d
    J[:, 0, 2] = camera.fx * x / d ** 2
    J[:, 1, 1] = -camera.fy / d
    J[:, 1, 2] = -camera.fy * y / d ** 2
    Sigma_2d = J @ Sigma_cam @ J.transpose(1, 2)                  # [N,2,2]
    Sigma_2d = Sigma_2d + eps * torch.eye(2, device=device)       # stabilizer

    u = camera.fx * x / d + camera.cx
    v = camera.cy - camera.fy * y / d
    order = torch.argsort(d[visible])           # nearest first for front-to-back
    idx = torch.where(visible)[0][order]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid = torch.stack([xs, ys], dim=-1).float()  # [H,W,2], pixel (u,v) coords

    image = torch.zeros(H, W, 3, device=device)
    T = torch.ones(H, W, 1, device=device)        # remaining transmittance
    for i in idx:
        delta = grid - torch.stack([u[i], v[i]])                 # [H,W,2]
        inv = torch.inverse(Sigma_2d[i])                         # [2,2]
        w = torch.exp(-0.5 * (delta @ inv * delta).sum(-1))      # 2D Gaussian
        a = (op[i] * w).unsqueeze(-1)                            # [H,W,1] alpha
        image = image + T * a * colors[i]
        T = T * (1 - a)
    return image + T * torch.ones(3, device=device)               # white background


if __name__ == "__main__":
    R, t = look_at(eye=(0.0, 0.0, 3.0), center=(0.0, 0.0, 0.0))
    cam = PinholeCamera(R, t, fx=64.0, fy=64.0, cx=32.0, cy=32.0, H=64, W=64)
    means = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.7, 0.0]],
                         dtype=torch.float32, requires_grad=True)
    scales = torch.full((3, 3), 0.25)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    opacity = torch.full((3, 1), 0.9)
    colors = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])

    _, pix = cam.project(means.detach())
    print(f"projected pixels (u,v): {pix.tolist()}  # left-red, right-green, top-blue")
    img = render(cam, means, scales, quats, opacity, colors)
    print(f"shape={tuple(img.shape)} finite={torch.isfinite(img).all().item()} "
          f"min={img.min():.4f} max={img.max():.4f} mean={img.mean():.4f}")
    for k, name in enumerate(["red-left", "green-right", "blue-top"]):
        uu, vv = int(pix[k, 0].round().clamp(0, 63)), int(pix[k, 1].round().clamp(0, 63))
        print(f"{name} @ (u={uu},v={vv}): rgb={img[vv, uu].tolist()}")
    img.mean().backward()
    print(f"backward ok: means.grad is None? {means.grad is None}")
    try:
        from PIL import Image
        Image.fromarray((img.detach().clamp(0, 1).numpy() * 255).astype("uint8")).save("/tmp/light_render.png")
        print("saved /tmp/light_render.png")
    except ImportError:
        pass

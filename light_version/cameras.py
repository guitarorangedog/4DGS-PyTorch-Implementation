"""Minimal pinhole camera: world -> camera -> pixels.

Convention (written out so splatting later is unambiguous):
- World: y-up, arbitrary units.
- Camera space: x = right, y = up, z = backward.
  The camera looks along -z (standard vision / 3DGS convention).
  A point in front of the camera has z_cam < 0; depth = -z_cam > 0.
- Pixels: u grows right, v grows DOWN (image rows).
  u = fx * (x / depth) + cx,  v = cy - fy * (y / depth).
  So a world point to the right gives larger u;
  a world point above gives smaller v.

Shapes: points [N,3], R [3,3], t [3].
"""

import torch
from dataclasses import dataclass


def look_at(eye, center, up=(0.0, 1.0, 0.0)):
    """Build (R, t) so the camera at `eye` looks at `center`.

    forward = normalized(center - eye) is where the camera looks.
    Rows of R are the camera axes in world coords: [right, up, -forward].
    Maps world -> camera: x_cam = R @ x_world + t, with t = -R @ eye.
    """
    eye = torch.as_tensor(eye, dtype=torch.float32)
    center = torch.as_tensor(center, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)

    forward = center - eye
    forward = forward / forward.norm()          # camera viewing direction
    right = torch.cross(forward, up, dim=0)
    right = right / right.norm()                # camera x-axis (world coords)
    cam_up = torch.cross(right, forward, dim=0)  # camera y-axis (world coords)

    R = torch.stack([right, cam_up, -forward], dim=0)
    t = -R @ eye
    return R, t


@dataclass
class PinholeCamera:
    R: torch.Tensor  # [3,3] world -> camera rotation
    t: torch.Tensor  # [3] world -> camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    H: int
    W: int

    def world_to_cam(self, pts):
        """pts [N,3] world -> [N,3] camera (x right, y up, z backward)."""
        return pts @ self.R.T + self.t

    def project(self, pts):
        """pts [N,3] world -> (cam [N,3], pixels [N,2]). Assumes depth > 0."""
        cam = self.world_to_cam(pts)
        depth = -cam[:, 2:3]                    # positive in front
        xn = cam[:, 0:1] / depth                # normalized x
        yn = cam[:, 1:2] / depth                # normalized y (up positive)
        u = self.fx * xn + self.cx              # right -> larger u
        v = self.cy - self.fy * yn              # up -> smaller v (rows go down)
        return cam, torch.cat([u, v], dim=1)


if __name__ == "__main__":
    R, t = look_at(eye=(0.0, 0.0, 3.0), center=(0.0, 0.0, 0.0))
    cam = PinholeCamera(R, t, fx=64.0, fy=64.0, cx=32.0, cy=32.0, H=64, W=64)
    pts = torch.tensor([
        [0.0, 0.0, 0.0],   # image center
        [1.0, 0.0, 0.0],   # to the right
        [0.0, 1.0, 0.0],   # above
    ])
    cam_pts, pix = cam.project(pts)
    for i, label in enumerate(["center", "right ", "up   "]):
        print(f"{label}: cam={cam_pts[i].tolist()} pixel=(u={pix[i, 0]:.2f}, v={pix[i, 1]:.2f})")

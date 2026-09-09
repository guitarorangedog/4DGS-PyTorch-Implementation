"""D-NeRF / Blender dataset reader [INFRA].

Expected folder layout (per view ``file_path`` without extension)::

    <source>/
      transforms_train.json   # frames: [{file_path, transform_matrix [4,4], time}]
      transforms_test.json
      train/r_0.png  test/r_0.png  ...

Faithful to official ``scene/dataset_readers.py`` (``read_timeline``,
``readCamerasFromTransforms``, ``generateCamerasFromTransforms``,
``readNerfSyntheticInfo``):

- Timestamps: all raw ``frame["time"]`` values from train+test are sorted and
  normalized by the maximum (``time_norm = t / t_max`` in ``[0, 1]``).
  Normalized time is stored on :class:`Camera` and later feeds HexPlane.
- Extrinsics: ``matrix = inv(c2w)`` then ``R = -matrix[:3,:3]^T`` with the
  first column re-negated, ``T = -matrix[:3,3]`` (exact official formula).
- Intrinsics: ``FoVx`` from ``camera_angle_x`` (fallback ``fl_x``/``w``);
  ``FoVy`` via focal round-trip at ``(W, H)``. Reuses ``gaussians.geometry``.
- Images: RGBA composited over white (default) / black, loaded at **native**
  resolution (see deviations).
- Video cameras: 160 spherical poses (radius 4.0, ``phi=-30``), times
  ``linspace(0, t_max)/t_max``; no images attached (see deviations).
- Point cloud: ``fused.ply`` when present, else 2000 uniform cube points
  (see deviations).
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from gaussians.geometry import focal2fov, fov2focal
from gaussians.sh import sh_to_rgb

__all__ = [
    "TRAIN_JSON",
    "TEST_JSON",
    "NUM_RANDOM_POINTS",
    "RANDOM_BOUND",
    "read_timeline",
    "c2w_to_R_T",
    "composite_rgba",
    "read_cameras_from_transforms",
    "generate_video_cameras",
    "load_or_sample_pointcloud",
]

TRAIN_JSON = "transforms_train.json"
TEST_JSON = "transforms_test.json"
NUM_RANDOM_POINTS = 2000
RANDOM_BOUND = 1.3


def read_timeline(source_path: str) -> tuple[dict, float]:
    """Map every raw timestamp to its ``[0, 1]`` normalized value.

    Mirrors official ``read_timeline``: union of train+test ``frame["time"]``
    values, sorted, divided by the maximum.

    Returns:
        ``(mapper, max_time)`` with ``mapper[raw] = raw / max_time``.
    """
    times: set = set()
    for name in (TRAIN_JSON, TEST_JSON):
        with open(os.path.join(source_path, name)) as f:
            contents = json.load(f)
        times.update(frame["time"] for frame in contents["frames"])
    ordered = sorted(times)
    max_time = float(max(ordered))
    return {t: float(t) / max_time for t in ordered}, max_time


def c2w_to_R_T(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a Blender ``c2w`` matrix to the stored ``(R, T)`` convention.

    Exact official formula (``readCamerasFromTransforms``)::

        matrix = inv(c2w);  R = -matrix[:3,:3]^T;  R[:,0] *= -1;  T = -matrix[:3,3]

    Args:
        c2w: ``[4, 4]`` camera-to-world matrix from the JSON frame.

    Returns:
        ``(R [3, 3], T [3])`` ready for :class:`Camera`.
    """
    matrix = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    R = -matrix[:3, :3].transpose()
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return np.float32(R), np.float32(T)


def composite_rgba(image: Image.Image, white_background: bool = True) -> np.ndarray:
    """Composite an RGBA image over a white/black background to RGB.

    Same as official: ``rgb * a + bg * (1 - a)`` in ``[0, 1]``.

    Returns:
        ``[H, W, 3]`` float64 array in ``[0, 1]``.
    """
    rgba = np.array(image.convert("RGBA"), dtype=np.float64) / 255.0
    bg = np.array([1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0])
    return rgba[..., :3] * rgba[..., 3:4] + bg * (1.0 - rgba[..., 3:4])


def _fovx_from_contents(contents: dict) -> float:
    if "camera_angle_x" in contents:
        return float(contents["camera_angle_x"])
    return focal2fov(float(contents["fl_x"]), float(contents["w"]))


def read_cameras_from_transforms(source_path: str, filename: str, white_background: bool,
                                 time_mapper: dict, extension: str = ".png"):
    """Read one split of frames into :class:`View` objects (frame order kept).

    Mirrors official ``readCamerasFromTransforms`` (same R/T/time/FoV math).

    Returns:
        List of :class:`View` (see ``data/scene.py``) with RGB ``[3, H, W]``
        float32 images in ``[0, 1]``.
    """
    from data.scene import View
    from data.cameras import Camera

    with open(os.path.join(source_path, filename)) as f:
        contents = json.load(f)
    fovx = _fovx_from_contents(contents)
    views = []
    for idx, frame in enumerate(contents["frames"]):
        rel = frame["file_path"] + extension
        image_path = os.path.join(source_path, rel)
        rgb = composite_rgba(Image.open(image_path), white_background)
        image = torch.from_numpy(rgb).float().permute(2, 0, 1)
        H, W = rgb.shape[:2]
        R, T = c2w_to_R_T(frame["transform_matrix"])
        fovy = focal2fov(fov2focal(fovx, W), H)
        camera = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy,
                        image_width=W, image_height=H,
                        time=time_mapper[frame["time"]])
        views.append(View(camera=camera, image=image,
                          image_name=os.path.splitext(os.path.basename(rel))[0]))
    return views


def generate_video_cameras(source_path: str, max_time: float, num_poses: int = 160,
                           extension: str = ".png"):
    """Generate spherical video poses (official ``generateCamerasFromTransforms``).

    160 cameras on a sphere (radius 4.0, ``phi=-30`` deg), times
    ``linspace(0, max_time) / max_time``. ``image`` is ``None`` (deviation,
    documented in ``data/scene.py``): video frames are render-only outputs.
    """
    from data.scene import View
    from data.cameras import Camera

    with open(os.path.join(source_path, TRAIN_JSON)) as f:
        template = json.load(f)
    fovx = _fovx_from_contents(template)
    # Video poses need an image size for FoVy; probe the template's first
    # frame header (lazy, no pixel load). D-NeRF is 800x800 square, where
    # the focal round-trip gives fovy == fovx exactly.
    first_rel = template["frames"][0]["file_path"] + extension
    with Image.open(os.path.join(source_path, first_rel)) as probe:
        W, H = probe.size
    fovy = focal2fov(fov2focal(fovx, W), H)

    def pose_spherical(theta_deg: float, phi_deg: float, radius: float) -> torch.Tensor:
        theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[2, 3] = radius  # trans_t(radius)
        rot_phi = torch.tensor([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float64)
        rot_theta = torch.tensor([
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ], dtype=torch.float64)
        base = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                            dtype=torch.float64)
        return base @ rot_theta @ rot_phi @ c2w

    views = []
    thetas = np.linspace(-180.0, 180.0, num_poses + 1)[:-1]
    times = np.linspace(0.0, max_time, num_poses) / max_time
    for idx, (theta, t) in enumerate(zip(thetas, times)):
        R, T = c2w_to_R_T(pose_spherical(float(theta), -30.0, 4.0).numpy())
        camera = Camera(R=R, T=T, FoVx=fovx, FoVy=fovy,
                        image_width=W, image_height=H, time=float(t))
        views.append(View(camera=camera, image=None, image_name=f"video_{idx:05d}"))
    return views


def load_or_sample_pointcloud(source_path: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``fused.ply`` if present, else sample the official random cube.

    Fallback mirrors official ``readNerfSyntheticInfo``: 2000 points uniform
    in ``[-1.3, 1.3]^3`` with near-gray colors (``SH2RGB(rand / 255)``).
    Uses a seeded RNG (deviation: official is unseeded).

    Returns:
        ``(points [N, 3], colors [N, 3])`` float32 torch tensors.
    """
    ply_path = os.path.join(source_path, "fused.ply")
    if os.path.exists(ply_path):
        from plyfile import PlyData

        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        points = np.vstack([v["x"], v["y"], v["z"]]).T
        if {"red", "green", "blue"} <= set(p.name for p in v.properties):
            colors = np.vstack([v["red"], v["green"], v["blue"]]).T / 255.0
        else:
            colors = np.full_like(points, 0.5)
        return torch.from_numpy(np.asarray(points, dtype=np.float32)), \
            torch.from_numpy(np.asarray(colors, dtype=np.float32))
    rng = np.random.default_rng(seed)
    xyz = (rng.random((NUM_RANDOM_POINTS, 3)).astype(np.float32) * 2 * RANDOM_BOUND - RANDOM_BOUND)
    shs = torch.from_numpy(rng.random((NUM_RANDOM_POINTS, 3)).astype(np.float32)) / 255.0
    colors = sh_to_rgb(shs)
    return torch.from_numpy(xyz), colors.float()

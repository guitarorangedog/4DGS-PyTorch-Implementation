"""DyNeRF / Neural-3D-Video dataset reader [INFRA].

Expected layout (post ``scripts/preprocess_dynerf.py`` frame extraction)::

    <source>/
      poses_bounds.npy            # [N_cams, 3, 5] LLFF poses + [N_cams, 2] near/far
      cam00/images/0000.png ...   # pre-extracted frames (sorted)
      cam01/images/...
      points3D_downsample2.ply    # COLMAP dense cloud, downsampled (REQUIRED)

Faithful to official ``scene/neural_3D_dataset_NDC.py``
(``Neural3D_NDC_Dataset``, downsample 1.0) +
``scene/dataset_readers.py`` (``readdynerfInfo``, ``format_infos``,
``format_render_poses``):

- Poses: LLFF ``[R|t|H,W,focal]`` rows; axis shuffle
  ``[col1, -col0, col2:4]``; per-frame stored pair
  ``R = -pose[:3,:3]; R[:,0] *= -1; T = -pose[:3,3] @ R``
  (identical formula in ``load_images_path`` and ``format_render_poses``).
- Split: ``eval_index = 0`` — train = every camera EXCEPT ``cam{eval}``,
  test = that camera's frames only. Physical camera index and temporal
  frame index are never conflated: ``(cam, frame)`` pairs carry one shared
  timestamp each.
- Time: ``frame_position_in_sorted_list / 300`` (constant ``countss``),
  in ``[0, 1)``. Later feeds HexPlane.
- FoV: ``FovX = focal2fov(f, H)``, ``FovY = focal2fov(f, W)`` — yes,
  swapped; ``format_infos`` reads ``image.shape[1]`` (= H of the ``[C,H,W]``
  tensor) for ``FovX``. Reproduced bug-compatibly (``format_render_poses``
  uses the unswapped order); flagged loudly.
- Video: 300-pose NeRF spiral (``average_poses``/``render_path_spiral``/
  ``viewmatrix`` ported verbatim, pure NumPy), times ``i/300``.
- Point cloud: ``points3D_downsample2.ply`` with ``red/green/blue`` attrs
  (official ``fetchPly`` crashes without it; we raise an explanatory error).
- Background: opaque RGB, official ``white_bg = False`` (black rasterizer bg).
- Resolution: ``img_wh = (1352, 1014)`` at downsample 1.0, LANCZOS resample.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image

from gaussians.geometry import focal2fov

__all__ = [
    "POSES_BOUNDS",
    "POINTS_PLY",
    "FRAME_COUNT",
    "llff_pose_to_R_T",
    "read_poses_bounds",
    "dynerf_split",
    "average_poses",
    "render_path_spiral",
    "spiral_video_poses",
    "read_dyacamera_views",
    "read_dynerf_video_views",
    "load_dynerf_pointcloud",
]

POSES_BOUNDS = "poses_bounds.npy"
POINTS_PLY = "points3D_downsample2.ply"
#: Official constant frame count per camera (``countss``); time denominator.
FRAME_COUNT = 300
#: Official working resolution (HARDCODED in ``Neural3D_NDC_Dataset`` for the
#: ``downsample=1.0`` call ``readdynerfInfo`` makes — independent of input).
IMG_WH = (1352, 1014)
#: The constructor then recomputes ``downsample = 2704 / img_wh[0] = 2.0``,
#: so the effective focal is ``raw_focal / 2`` (official quirk, reproduced).
EFFECTIVE_DOWNSAMPLE = 2704 / IMG_WH[0]


def llff_pose_to_R_T(pose34: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LLFF ``[3, 4]`` pose (post axis-shuffle) -> stored ``(R, T)``.

    Exact official formula (``load_images_path`` / ``format_render_poses``)::

        R = -pose[:3,:3];  R[:, 0] *= -1;  T = -pose[:3,3] @ R
    """
    R = -np.asarray(pose34, dtype=np.float64)[:3, :3]
    R[:, 0] = -R[:, 0]
    T = -np.asarray(pose34, dtype=np.float64)[:3, 3].dot(R)
    return np.float32(R), np.float32(T)


def read_poses_bounds(source_path: str):
    """Parse ``poses_bounds.npy`` (official ``load_meta`` pose preamble).

    Returns:
        ``(poses [N,3,4] axis-shuffled, effective focal, IMG_WH,
        near_fars [N,2])``. The focal is the raw LLFF focal divided by the
        effective downsample (2.0); the working resolution is always
        ``IMG_WH`` — see module constants.
    """
    path = os.path.join(source_path, POSES_BOUNDS)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DyNeRF calibration {path} not found.")
    poses_arr = np.load(path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5])
    near_fars = poses_arr[:, -2:]
    _, _, raw_focal = poses[0, :, -1]
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    return (poses.astype(np.float64), float(raw_focal) / EFFECTIVE_DOWNSAMPLE,
            IMG_WH, near_fars)


def _cam_frame_files(source_path: str, cam_index: int) -> list[str]:
    videos = sorted(glob.glob(os.path.join(source_path, "cam*.mp4")))
    if cam_index >= len(videos):
        raise FileNotFoundError(
            f"Camera {cam_index}: expected video {os.path.join(source_path, 'cam*')} "
            f"layout (found {len(videos)} cam videos).")
    img_dir = os.path.join(videos[cam_index].split(".")[0], "images")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"Camera {cam_index}: pre-extracted frames {img_dir}/ not found. "
            "Run the official frame-extraction preprocess first "
            "(scripts/preprocess_dynerf.py); this reader does not decode mp4.")
    files = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                   if f.lower().endswith(".png"))
    if not files:
        raise FileNotFoundError(f"No frames in {img_dir}/.")
    return files


def dynerf_split(n_cams: int, eval_index: int = 0) -> tuple[list[int], list[int]]:
    """Official train/test camera split (``load_images_path`` logic).

    Train = all cameras except ``eval_index``; test = that camera only.
    """
    train = [i for i in range(n_cams) if i != eval_index]
    return train, [eval_index]


def average_poses(poses: np.ndarray) -> np.ndarray:
    """NeRF pose averaging (official ``average_poses``, verbatim)."""
    center = poses[..., 3].mean(0)
    z = poses[..., 2].mean(0)
    z = z / np.linalg.norm(z)
    y_ = poses[..., 1].mean(0)
    x = np.cross(z, y_)
    x = x / np.linalg.norm(x)
    y = np.cross(x, z)
    return np.stack([x, y, z, center], 1)


def render_path_spiral(c2w: np.ndarray, up: np.ndarray, rads: np.ndarray,
                       focal: float, zdelta: float, zrate: float = 0.5,
                       N_rots: int = 2, N: int = 120) -> list[np.ndarray]:
    """NeRF spiral trajectory (official ``render_path_spiral``, verbatim)."""
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4],
                   np.array([np.cos(theta), -np.sin(theta),
                             -np.sin(theta * zrate), 1.0]) * rads)
        z = c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
        z = z / np.linalg.norm(z)
        vec2 = z / np.linalg.norm(z)
        vec0 = np.cross(up, vec2)
        vec0 = vec0 / np.linalg.norm(vec0)
        vec1 = np.cross(vec2, vec0)
        vec1 = vec1 / np.linalg.norm(vec1)
        m = np.eye(4)
        m[:3] = np.stack([-vec0, vec1, vec2, c], 1)
        render_poses.append(m)
    return render_poses


def spiral_video_poses(poses_all: np.ndarray, near_fars: np.ndarray,
                       N_views: int = 300) -> np.ndarray:
    """Official ``get_spiral`` validation trajectory (verbatim math)."""
    c2w = average_poses(poses_all)
    up = poses_all[:, :3, 1].sum(0)
    up = up / np.linalg.norm(up)
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    zdelta = near_fars.min() * 0.2
    tt = poses_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    return np.stack(render_path_spiral(c2w, up, rads, focal, zdelta,
                                       zrate=0.5, N=N_views))


def _load_frame(path: str, img_wh: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize(img_wh, Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def read_dyacamera_views(source_path: str, split: str, eval_index: int = 0):
    """Read train/test ``(cam, frame)`` views (official ``load_images_path``).

    Frame images stay on CPU as ``[3, H, W]`` float32 ``[0, 1]``.
    Returns a list of :class:`View` in ``(cam, frame)`` order.
    """
    from data.cameras import Camera
    from data.scene import View

    assert split in ("train", "test")
    poses, focal, img_wh, _ = read_poses_bounds(source_path)
    W, H = img_wh
    n_cams = poses.shape[0]
    train_cams, test_cams = dynerf_split(n_cams, eval_index)
    wanted = train_cams if split == "train" else test_cams
    views = []
    for cam in wanted:
        for frame_pos, path in enumerate(_cam_frame_files(source_path, cam)[:FRAME_COUNT]):
            R, T = llff_pose_to_R_T(poses[cam])
            # Bug-compatible with format_infos: FovX from H, FovY from W.
            views.append(View(
                camera=Camera(R=R, T=T, FoVx=focal2fov(focal, H),
                              FoVy=focal2fov(focal, W),
                              image_width=W, image_height=H,
                              time=frame_pos / FRAME_COUNT),
                image=_load_frame(path, img_wh),
                image_name=f"cam{cam:02d}_{frame_pos:04d}"))
    if not views:
        raise FileNotFoundError(f"DyNeRF {split} split is empty in {source_path}.")
    return views


def read_dynerf_video_views(source_path: str, N_views: int = 300):
    """Spiral validation trajectory (official ``val_poses`` + ``format_render_poses``).

    Render-only: ``image=None`` (official reuses the first train image; we
    attach nothing — same deviation as the D-NeRF reader).
    """
    from data.cameras import Camera
    from data.scene import View

    poses, focal, img_wh, near_fars = read_poses_bounds(source_path)
    W, H = img_wh
    val_poses = spiral_video_poses(poses, near_fars, N_views=N_views)
    views = []
    for idx, p in enumerate(val_poses):
        R, T = llff_pose_to_R_T(p[:3, :])
        views.append(View(
            camera=Camera(R=R, T=T, FoVx=focal2fov(focal, W),
                          FoVy=focal2fov(focal, H),
                          image_width=W, image_height=H,
                          time=idx / len(val_poses)),
            image=None, image_name=f"video_{idx:05d}"))
    return views


def load_dynerf_pointcloud(source_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``points3D_downsample2.ply`` (official ``fetchPly`` contract).

    Returns ``(points [N,3], colors [N,3])`` float32; colors from
    ``red/green/blue`` uint8 attrs.
    """
    from plyfile import PlyData

    ply_path = os.path.join(source_path, POINTS_PLY)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"DyNeRF point cloud {ply_path} not found. Generate it with the "
            "official COLMAP pipeline (colmap.sh) and downsample to "
            f"{POINTS_PLY} first.")
    v = PlyData.read(ply_path)["vertex"]
    names = {p.name for p in v.properties}
    if {"red", "green", "blue"} - names:
        raise ValueError(f"{ply_path} lacks red/green/blue color attributes.")
    points = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    colors = (np.vstack([v["red"], v["green"], v["blue"]]).T / 255.0).astype(np.float32)
    return torch.from_numpy(points), torch.from_numpy(colors)

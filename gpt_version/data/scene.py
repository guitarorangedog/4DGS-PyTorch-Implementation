"""Minimal Scene assembly [INFRA].

Holds what static training (Commit 9) and rendering (Commit 10) need:

- ``train_views`` / ``test_views`` / ``video_views``: ordered lists of
  :class:`View` (``camera`` + ``image`` + ``image_name``). ``Camera`` itself
  stays image-free (Commit 3); images travel alongside in the view.
- ``points`` / ``colors``: ``[N, 3]`` float32 init cloud for
  ``CanonicalGaussianModel.create_from_pointcloud``.
- ``scene_extent``: NeRF++-style radius (drives densification thresholds).
- ``aabb_min`` / ``aabb_max``: ``[3]`` point-cloud bounds (later: HexPlane
  ``set_aabb``).
- ``maxtime``: maximum raw timestamp (informational; normalized time lives
  on each camera).

Dispatch is intentionally minimal: only D-NeRF (``transforms_train.json``)
is enabled in this commit (Commit 7). DyNeRF/HyperNeRF arrive in Commit 18.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from data.cameras import Camera

__all__ = ["View", "Scene", "get_nerfpp_norm", "load_scene"]


@dataclass
class View:
    """One dataset view: camera geometry + pixels.

    Attributes:
        camera: :class:`Camera` with normalized ``time`` in ``[0, 1]``.
        image: ``[3, H, W]`` float32 RGB in ``[0, 1]``, or ``None`` for
            render-only video poses (official reuses the first train image
            there; we attach nothing instead — deviation documented in
            ``data/dnerf.py`` — since video frames are outputs, not targets).
        image_name: stem identifier, e.g. ``"train_r_0"`` / ``"video_00042"``.
    """

    camera: Camera
    image: torch.Tensor | None
    image_name: str


def get_nerfpp_norm(views: list[View]) -> tuple[np.ndarray, float]:
    """NeRF++-style normalization from camera centers (official ``getNerfppNorm``).

    ``center = mean(centers)``; ``radius = max||c - center|| * 1.1``;
    ``translate = -center``. Uses :attr:`Camera.camera_center`, which Commit 3
    verified against the official ``getWorld2View2`` path to ``~1e-7``.

    Returns:
        ``(translate [3], radius float)``; radius is the ``scene_extent``.
    """
    centers = np.stack([v.camera.camera_center.detach().cpu().numpy() for v in views], axis=0)
    center = centers.mean(axis=0)
    radius = float(np.linalg.norm(centers - center, axis=1).max() * 1.1)
    return -center, radius


@dataclass
class Scene:
    """Loaded dataset ready for training/rendering."""

    source_path: str
    dataset_type: str
    train_views: list[View] = field(default_factory=list)
    test_views: list[View] = field(default_factory=list)
    video_views: list[View] = field(default_factory=list)
    points: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))
    colors: torch.Tensor = field(default_factory=lambda: torch.empty(0, 3))
    scene_extent: float = 1.0
    aabb_min: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    aabb_max: torch.Tensor = field(default_factory=lambda: torch.zeros(3))
    maxtime: float = 1.0


#: Sentinel files for dataset auto-detection, in official ``Scene`` order.
DATASET_SENTINELS = (
    ("dnerf", "transforms_train.json"),
    ("dynerf", "poses_bounds.npy"),
    ("hypernerf", "dataset.json"),
)

#: Rasterizer background default per dataset (official ``white_bg`` flags).
DATASET_BG = {"dnerf": True, "dynerf": False, "hypernerf": True}

#: Extra informational time denominators (official ``maxtime`` values).
DATASET_MAXTIME = {"dynerf": 300, "hypernerf": 1.0}


def detect_dataset(source_path: str) -> str:
    """Auto-detect dataset family by sentinel files (official precedence).

    Raises:
        FileNotFoundError: no known layout (also covers official-only
            layouts we do not support: ``sparse/`` COLMAP,
            ``train_meta.json`` Panoptic, ``points3D_multipleview.ply``).
    """
    for dataset_type, sentinel in DATASET_SENTINELS:
        if os.path.exists(os.path.join(source_path, sentinel)):
            return dataset_type
    raise FileNotFoundError(
        f"No supported dataset layout in {source_path} "
        f"(looked for {[s for _, s in DATASET_SENTINELS]}). "
        "COLMAP/Panoptic/MultipleView layouts are not supported.")


def load_scene(source_path: str, white_background: bool | None = None,
               eval_mode: bool = True, extension: str = ".png", seed: int = 0,
               dataset_type: str | None = None, eval_index: int = 0,
               ratio: float = 0.5) -> Scene:
    """Load a dataset folder into a :class:`Scene`.

    Args:
        source_path: dataset root.
        white_background: rasterizer background; ``None`` selects the
            per-dataset official default (D-NeRF white, DyNeRF black,
            HyperNeRF white).
        eval_mode: with ``False``, test views merge into train (D-NeRF
            official behavior; DyNeRF/HyperNeRF keep their fixed splits).
        extension: D-NeRF image extension.
        seed: RNG seed for synthetic fallback clouds.
        dataset_type: ``"dnerf"`` / ``"dynerf"`` / ``"hypernerf"``;
            ``None`` auto-detects by sentinel files.
        eval_index: DyNeRF held-out camera (official default 0).
        ratio: HyperNeRF resolution ratio (official call uses 0.5 -> ``2x``).

    Raises:
        ValueError: explicit ``dataset_type`` contradicts auto-detection.
    """
    from data import dnerf

    detected = detect_dataset(source_path)
    if dataset_type is None:
        dataset_type = detected
    elif dataset_type != detected:
        raise ValueError(
            f"Requested dataset_type={dataset_type!r} but {source_path} "
            f"detects as {detected!r}.")
    if white_background is None:
        white_background = DATASET_BG[dataset_type]

    if dataset_type == "dnerf":
        time_mapper, max_time = dnerf.read_timeline(source_path)
        train_views = dnerf.read_cameras_from_transforms(
            source_path, dnerf.TRAIN_JSON, white_background, time_mapper, extension)
        test_views = dnerf.read_cameras_from_transforms(
            source_path, dnerf.TEST_JSON, white_background, time_mapper, extension)
        if not eval_mode:
            train_views.extend(test_views)
            test_views = []
        video_views = dnerf.generate_video_cameras(source_path, max_time, extension=extension)
        points, colors = dnerf.load_or_sample_pointcloud(source_path, seed=seed)
    elif dataset_type == "dynerf":
        from data import dynerf

        train_views = dynerf.read_dyacamera_views(source_path, "train", eval_index)
        test_views = dynerf.read_dyacamera_views(source_path, "test", eval_index)
        video_views = dynerf.read_dynerf_video_views(source_path)
        points, colors = dynerf.load_dynerf_pointcloud(source_path)
        max_time = DATASET_MAXTIME["dynerf"]
    elif dataset_type == "hypernerf":
        from data import hypernerf

        train_views = hypernerf.read_hyper_views(source_path, "train", ratio)
        test_views = hypernerf.read_hyper_views(source_path, "test", ratio)
        video_views = hypernerf.read_hyper_video_views(source_path, ratio)
        points, colors = hypernerf.load_hyper_pointcloud(source_path)
        max_time = DATASET_MAXTIME["hypernerf"]
    else:  # pragma: no cover - detect_dataset restricts the values
        raise ValueError(f"Unsupported dataset_type={dataset_type!r}.")

    _, radius = get_nerfpp_norm(train_views)
    return Scene(
        source_path=os.path.abspath(source_path),
        dataset_type=dataset_type,
        train_views=train_views,
        test_views=test_views,
        video_views=video_views,
        points=points,
        colors=colors,
        scene_extent=radius,
        aabb_min=points.min(dim=0).values,
        aabb_max=points.max(dim=0).values,
        maxtime=max_time,
    )

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


def load_scene(source_path: str, white_background: bool = True, eval_mode: bool = True,
               extension: str = ".png", seed: int = 0) -> Scene:
    """Load a dataset folder into a :class:`Scene` (D-NeRF only for now).

    Mirrors official ``readNerfSyntheticInfo`` flow: timeline -> train/test
    views -> video views -> normalization -> point cloud. With
    ``eval_mode=False``, test views merge into train (official behavior).

    Raises:
        FileNotFoundError: if ``transforms_train.json`` is absent (only
            D-NeRF is wired in Commit 7).
    """
    from data import dnerf

    train_json = os.path.join(source_path, dnerf.TRAIN_JSON)
    if not os.path.exists(train_json):
        raise FileNotFoundError(
            f"No {dnerf.TRAIN_JSON} in {source_path}: only D-NeRF datasets "
            "are supported as of Commit 7."
        )
    time_mapper, max_time = dnerf.read_timeline(source_path)
    train_views = dnerf.read_cameras_from_transforms(
        source_path, dnerf.TRAIN_JSON, white_background, time_mapper, extension)
    test_views = dnerf.read_cameras_from_transforms(
        source_path, dnerf.TEST_JSON, white_background, time_mapper, extension)
    if not eval_mode:
        train_views.extend(test_views)
        test_views = []
    video_views = dnerf.generate_video_cameras(source_path, max_time, extension=extension)

    _, radius = get_nerfpp_norm(train_views)
    points, colors = dnerf.load_or_sample_pointcloud(source_path, seed=seed)

    return Scene(
        source_path=os.path.abspath(source_path),
        dataset_type="dnerf",
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

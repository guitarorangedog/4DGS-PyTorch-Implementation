"""HyperNeRF dataset reader [INFRA].

Expected layout (per ``scene/hyper_loader.py::Load_hyper_data``)::

    <source>/
      scene.json        # {near, far, scale, center}
      metadata.json     # {id: {camera_id, warp_id}}
      dataset.json      # {ids, val_ids, train_ids}
      camera/{id}.json  # {orientation [3,3], position [3], focal_length, ...}
      rgb/{R}x/{id}.png  # R = int(1/ratio); official call uses ratio=0.5 -> 2x
      points3D_downsample2.ply   # (REQUIRED, same fetchPly contract as DyNeRF)

Faithful behavior:

- Split: ``val_ids`` empty -> ``i_train = every 4th``, ``i_test = i_train+2``
  (last dropped); else membership in ``train_ids``/``val_ids``.
- Time: ``warp_id / max(warp_id)`` in ``[0, 1]`` (official ``all_time``).
- Camera: ``R = orientation.T``; ``T = -position @ R``; ``FoVx/y`` from
  ``focal_length`` with the first camera's ``(h, w)`` for every view
  (official ``data_class.h/w``). ``image_size`` JSON order is ``[W, H]``
  (``image_size_x = [0]``); pixel dims still come from the PNG itself.
- Video: official ``video = deepcopy(test)`` — our ``video_views`` are the
  test views (render-only copies with ``image=None``).
- Point cloud: ``points3D_downsample2.ply`` REQUIRED (official crashes
  without it; we raise an explanatory error).
- Images: opaque RGB, loaded at file resolution, no alpha compositing.
- Masks (``covisible/``): parsed by official test path but never consumed by
  rendering/training; omitted here (documented).
- ``scene.json`` near/far/scale/center are recorded on the Scene for
  provenance but the official path does not rescale cameras with them;
  ``getNerfppNorm`` over train views provides the extent (as official).
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from gaussians.geometry import focal2fov

__all__ = [
    "POINTS_PLY",
    "hyper_split",
    "read_hyper_meta",
    "hyper_camera_to_R_T",
    "read_hyper_views",
    "read_hyper_video_views",
    "load_hyper_pointcloud",
]

POINTS_PLY = "points3D_downsample2.ply"


def read_hyper_meta(source_path: str, ratio: float = 0.5) -> dict:
    """Load and normalize ``scene/dataset/metadata.json`` (official ``__init__``).

    Returns a dict with ``ids, train_idx, test_idx, times [0,1],
    max_time, min_time, rgb_dir, scene (near/far/scale/center)``.
    """
    for name in ("scene.json", "metadata.json", "dataset.json"):
        if not os.path.exists(os.path.join(source_path, name)):
            raise FileNotFoundError(
                f"HyperNeRF metadata {os.path.join(source_path, name)} not found.")
    with open(os.path.join(source_path, "scene.json")) as f:
        scene_json = json.load(f)
    with open(os.path.join(source_path, "metadata.json")) as f:
        meta_json = json.load(f)
    with open(os.path.join(source_path, "dataset.json")) as f:
        dataset_json = json.load(f)

    ids = list(dataset_json["ids"])
    warp = np.array([meta_json[i]["warp_id"] for i in ids], dtype=np.float64)
    max_warp = warp.max()
    times = (warp / max_warp).tolist()
    return {
        "ids": ids,
        "meta": meta_json,
        "train_idx": None,  # resolved by hyper_split
        "test_idx": None,
        "times": times,
        "max_time": float(np.max(times)),
        "min_time": float(np.min(times)),
        "rgb_dir": os.path.join(source_path, "rgb", f"{int(1 / ratio)}x"),
        "camera_dir": os.path.join(source_path, "camera"),
        "scene": {k: scene_json[k] for k in ("near", "far", "scale", "center")},
        "val_ids": list(dataset_json.get("val_ids", [])),
        "train_ids": list(dataset_json.get("train_ids", [])),
    }


def hyper_split(ids: list, val_ids: list, train_ids: list) -> tuple[list[int], list[int]]:
    """Official train/test index split (``Load_hyper_data.__init__`` logic).

    Empty ``val_ids``: every 4th frame trains, ``+2`` offset tests (last
    dropped). Otherwise: membership in ``train_ids`` / ``val_ids``.
    """
    if len(val_ids) == 0:
        i_train = [i for i in range(len(ids)) if i % 4 == 0]
        i_test = [i + 2 for i in i_train][:-1]
        return i_train, [i for i in i_test if i < len(ids)]
    i_train, i_test = [], []
    for i, _id in enumerate(ids):
        if _id in val_ids:
            i_test.append(i)
        if _id in train_ids:
            i_train.append(i)
    return i_train, i_test


def hyper_camera_to_R_T(orientation: np.ndarray, position: np.ndarray):
    """Official HyperNeRF conversion (``load_raw`` / ``format_hyper_data``)::

        R = orientation.T;  T = -position @ R
    """
    R = np.asarray(orientation, dtype=np.float64).T
    T = -np.asarray(position, dtype=np.float64) @ R
    return np.float32(R), np.float32(T)


def _read_camera_json(camera_dir: str, _id: str) -> dict:
    path = os.path.join(camera_dir, f"{_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HyperNeRF camera {path} not found.")
    with open(path) as f:
        return json.load(f)


def _load_rgb(path: str) -> tuple[torch.Tensor, int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"HyperNeRF image {path} not found.")
    with Image.open(path) as img:
        img = img.convert("RGB")
        W, H = img.size
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1), W, H


def read_hyper_views(source_path: str, split: str, ratio: float = 0.5):
    """Read train/test views (official ``load_raw`` image path + metadata).

    FoV for every view uses the FIRST camera's ``(h, w)`` with its
    ``focal_length`` (official ``data_class.h/w``); pixel dims come from each
    PNG. Returns :class:`View` list in dataset order.
    """
    from data.cameras import Camera
    from data.scene import View

    assert split in ("train", "test")
    meta = read_hyper_meta(source_path, ratio)
    i_train, i_test = hyper_split(meta["ids"], meta["val_ids"], meta["train_ids"])
    wanted = i_train if split == "train" else i_test
    first_cam = _read_camera_json(meta["camera_dir"], meta["ids"][0])
    ref_h, ref_w = first_cam["image_size"][1], first_cam["image_size"][0]
    ref_focal = float(first_cam["focal_length"])
    views = []
    for uid, idx in enumerate(wanted):
        _id = meta["ids"][idx]
        cam = _read_camera_json(meta["camera_dir"], _id)
        image, W, H = _load_rgb(os.path.join(meta["rgb_dir"], f"{_id}.png"))
        R, T = hyper_camera_to_R_T(cam["orientation"], cam["position"])
        views.append(View(
            camera=Camera(R=R, T=T,
                          FoVx=focal2fov(ref_focal, ref_w),
                          FoVy=focal2fov(ref_focal, ref_h),
                          image_width=W, image_height=H,
                          time=meta["times"][idx]),
            image=image, image_name=_id))
    if not views:
        raise FileNotFoundError(f"HyperNeRF {split} split is empty in {source_path}.")
    return views


def read_hyper_video_views(source_path: str, ratio: float = 0.5):
    """Official video = test views (``deepcopy(test)``); render-only copies."""
    import dataclasses

    return [dataclasses.replace(v, image=None)
            for v in read_hyper_views(source_path, "test", ratio)]


def load_hyper_pointcloud(source_path: str):
    """Load ``points3D_downsample2.ply`` (same contract as DyNeRF)."""
    from data.dynerf import load_dynerf_pointcloud  # identical fetchPly contract

    try:
        return load_dynerf_pointcloud(source_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"HyperNeRF point cloud {os.path.join(source_path, POINTS_PLY)} "
            "not found (see COLMAP guidance in the official README).") from e

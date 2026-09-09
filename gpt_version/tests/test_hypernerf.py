"""CPU + parity tests for data/hypernerf.py. Run via `python3 -m tests.test_hypernerf`."""

import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from data.hypernerf import hyper_camera_to_R_T, hyper_split, read_hyper_meta
from data.scene import detect_dataset, load_scene


def _write_fixture(root: str) -> None:
    rng = np.random.default_rng(1)
    ids = [f"frame_{i:03d}" for i in range(5)]
    with open(os.path.join(root, "scene.json"), "w") as f:
        json.dump({"near": 0.1, "far": 10.0, "scale": 1.0,
                   "center": [0.0, 0.0, 0.0]}, f)
    meta = {i: {"camera_id": f"cam_{int(i[-1]) % 2}",
                "warp_id": w} for i, w in zip(ids, [0, 2, 4, 1, 3])}
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "dataset.json"), "w") as f:
        json.dump({"ids": ids, "val_ids": ["frame_001"],
                   "train_ids": ["frame_000", "frame_002", "frame_004"]}, f)
    os.makedirs(os.path.join(root, "camera"), exist_ok=True)
    os.makedirs(os.path.join(root, "rgb", "1x"), exist_ok=True)
    for k, i in enumerate(ids):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        with open(os.path.join(root, "camera", f"{i}.json"), "w") as f:
            json.dump({
                "orientation": Q.tolist(),
                "position": [0.2 * k, 0.1 * k, 0.05 * k],
                "focal_length": 500.0,
                "principal_point": [32.0, 24.0],
                "skew": 0.0,
                "pixel_aspect_ratio": 1.0,
                "radial_distortion": [0.0, 0.0, 0.0],
                "tangential_distortion": [0.0, 0.0],
                "image_size": [64, 48]}, f)
        arr = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr, "RGB").save(os.path.join(root, "rgb", "1x", f"{i}.png"))
    xyz = rng.random((40, 3)).astype(np.float32) * 2 - 1
    rgb = (rng.random((40, 3)) * 255).astype(np.float32)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"),
             ("nz", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")]
    el = np.empty(40, dtype=dtype)
    el[:] = list(map(tuple, np.concatenate([xyz, np.zeros_like(xyz), rgb], axis=1)))
    PlyData([PlyElement.describe(el, "vertex")]).write(
        os.path.join(root, "points3D_downsample2.ply"))


def _stub_heavy():
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    _tk = types.ModuleType("tkinter")
    _tk.W = None
    sys.modules.setdefault("tkinter", _tk)
    _sk = types.ModuleType("simple_knn")
    _sk_C = types.ModuleType("simple_knn._C")
    _sk_C.distCUDA2 = None
    _sk._C = _sk_C
    sys.modules.setdefault("simple_knn", _sk)
    sys.modules.setdefault("simple_knn._C", _sk_C)


def test_layout_split_time() -> None:
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        assert detect_dataset(root) == "hypernerf"
        # Pure split logic, both branches.
        assert hyper_split(["a", "b", "c", "d", "e", "f", "g", "h", "i"], [], []) == (
            [0, 4, 8], [2, 6])
        assert hyper_split(["a", "b", "c"], ["c"], ["a", "b"]) == ([0, 1], [2])
        meta = read_hyper_meta(root, ratio=1.0)
        assert meta["max_time"] == 1.0 and meta["min_time"] == 0.0
        assert meta["times"] == [0.0, 0.5, 1.0, 0.25, 0.75]
        scene = load_scene(root, seed=0, ratio=1.0)
        assert scene.dataset_type == "hypernerf" and scene.maxtime == 1.0
        assert [v.image_name for v in scene.train_views] == [
            "frame_000", "frame_002", "frame_004"]
        assert [v.image_name for v in scene.test_views] == ["frame_001"]
        assert [v.camera.time for v in scene.train_views] == [0.0, 1.0, 0.75]
        assert scene.test_views[0].camera.time == 0.5
        assert scene.train_views[0].image.shape == (3, 48, 64)
        # Video = test views (official deepcopy), render-only.
        assert [v.image_name for v in scene.video_views] == ["frame_001"]
        assert scene.video_views[0].image is None
        assert scene.points.shape == (40, 3)
        assert scene.scene_extent > 0


def test_official_parity() -> None:
    _stub_heavy()
    import scene.hyper_loader as official
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        ref_train = official.Load_hyper_data(root, 1.0, False, split="train")
        ref_test = official.Load_hyper_data(root, 1.0, False, split="test")
        scene = load_scene(root, seed=0, ratio=1.0)
        assert list(ref_train.i_train) == [0, 2, 4]
        assert list(ref_test.i_test) == [1]
        assert list(ref_train.all_time) == [0.0, 0.5, 1.0, 0.25, 0.75]
        for view, idx in zip(scene.train_views, [0, 2, 4]):
            raw = ref_train.load_raw(idx)
            assert np.allclose(view.camera.R.numpy(), raw.R, atol=1e-6)
            assert np.allclose(view.camera.T.numpy(), raw.T, atol=1e-6)
            assert abs(view.camera.FoVx - raw.FovX) < 1e-9
            assert abs(view.camera.FoVy - raw.FovY) < 1e-9
            assert view.camera.time == raw.time
            assert torch.allclose(view.image, raw.image, atol=1e-6)
            assert (view.camera.image_width, view.camera.image_height) == (64, 48)
        raw_test = ref_test.load_raw(1)
        assert scene.test_views[0].camera.time == raw_test.time == 0.5
        assert torch.allclose(scene.test_views[0].image, raw_test.image, atol=1e-6)
        print("test_official_parity: passed (split/time/R/T/FoV/images)")


def test_explicit_type_mismatch_errors() -> None:
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        try:
            load_scene(root, dataset_type="dnerf")
        except ValueError as e:
            assert "detects as" in str(e)
            return
    raise AssertionError("type mismatch should raise")


if __name__ == "__main__":
    test_layout_split_time()
    test_official_parity()
    test_explicit_type_mismatch_errors()
    print("test_hypernerf.py: all 3 tests passed")

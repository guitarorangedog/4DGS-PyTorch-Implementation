"""CPU tests for data/dnerf.py + data/scene.py. Run via `python3 -m tests.test_dnerf`."""

import importlib.util
import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data import dnerf
from data.scene import View, get_nerfpp_norm, load_scene


def _c2w(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> list:
    m = np.eye(4)
    m[:3, 3] = [tx, ty, tz]
    return m.tolist()


def _write_png(path: str, rgb: tuple, alpha: int, size: int = 800) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3] = *rgb, alpha
    Image.fromarray(arr, "RGBA").save(path)


def _fixture(root: str) -> None:
    train_frames = [
        {"file_path": f"train/r_{i}", "transform_matrix": _c2w(0.1 * i, 0, 0), "time": t}
        for i, t in enumerate([0, 1, 2])
    ]
    test_frames = [
        {"file_path": f"test/r_{i}", "transform_matrix": _c2w(0, 0.1 * i, 0), "time": t}
        for i, t in enumerate([0, 2])
    ]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    _write_png(os.path.join(root, "train", "r_0.png"), (255, 0, 0), 255)
    _write_png(os.path.join(root, "train", "r_1.png"), (0, 255, 0), 255)
    _write_png(os.path.join(root, "train", "r_2.png"), (0, 0, 255), 255)
    _write_png(os.path.join(root, "test", "r_0.png"), (0, 0, 0), 0)  # transparent -> white
    _write_png(os.path.join(root, "test", "r_1.png"), (0, 0, 0), 255)


def test_timeline_normalization() -> None:
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        mapper, maxtime = dnerf.read_timeline(root)
    assert maxtime == 2.0
    assert mapper == {0: 0.0, 1: 0.5, 2: 1.0}


def test_extrinsics_conversion_identity() -> None:
    R, T = dnerf.c2w_to_R_T(np.eye(4))
    assert np.allclose(R, np.diag([1.0, -1.0, -1.0]), atol=1e-6)
    assert np.allclose(T, np.zeros(3), atol=1e-6)


def test_scene_assembly() -> None:
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        scene = load_scene(root, white_background=True, eval_mode=True, seed=0)
    assert scene.dataset_type == "dnerf" and scene.maxtime == 2.0
    assert [v.image_name for v in scene.train_views] == ["r_0", "r_1", "r_2"]
    assert [v.image_name for v in scene.test_views] == ["r_0", "r_1"]
    assert [v.camera.time for v in scene.train_views] == [0.0, 0.5, 1.0]
    assert [v.camera.time for v in scene.test_views] == [0.0, 1.0]
    # Intrinsics: native 800x800, FoVx from JSON, FoVy round-trip.
    v0 = scene.train_views[0]
    assert v0.image.shape == (3, 800, 800)
    assert v0.camera.image_width == 800 and v0.camera.image_height == 800
    assert abs(v0.camera.FoVx - 0.9) < 1e-9
    # Colors: solid red frame; transparent test frame composited to white.
    assert torch.allclose(v0.image[:, 400, 400], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(
        scene.test_views[0].image[:, 400, 400], torch.ones(3), atol=1e-6)
    # Deterministic order + extent/AABB/point cloud invariants.
    assert scene.scene_extent > 0
    assert (scene.aabb_max > scene.aabb_min).all()
    assert scene.points.shape == (2000, 3) and scene.colors.shape == (2000, 3)
    assert ((scene.points >= -1.3) & (scene.points <= 1.3)).all()
    assert len(scene.video_views) == 160
    assert scene.video_views[0].image is None
    assert abs(scene.video_views[0].camera.time - 0.0) < 1e-9
    assert abs(scene.video_views[-1].camera.time - 1.0) < 1e-9


def test_eval_false_merges_test_into_train() -> None:
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        scene = load_scene(root, eval_mode=False, seed=0)
    assert len(scene.train_views) == 5 and scene.test_views == []


def test_official_parity() -> None:
    import sys
    import types
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    # Stub heavy/compiled third-party deps that the official D-NeRF reader
    # path never calls (import-time only, via scene/__init__ chain).
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
    spec = importlib.util.spec_from_file_location(
        "official_readers", "/tmp/opencode/4DGaussians/scene/dataset_readers.py")
    official = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(official)
    except ModuleNotFoundError as e:
        print(f"test_official_parity: skipped ({e})")
        return
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        np.random.seed(0)
        ref = official.readNerfSyntheticInfo(root, True, True, ".png")
        scene = load_scene(root, white_background=True, eval_mode=True, seed=0)
    for ours, theirs in zip(scene.train_views + scene.test_views,
                            ref.train_cameras + ref.test_cameras):
        assert np.allclose(ours.camera.R.numpy(), theirs.R, atol=1e-5)
        assert np.allclose(ours.camera.T.numpy(), theirs.T, atol=1e-5)
        assert abs(ours.camera.time - theirs.time) < 1e-9
        assert abs(ours.camera.FoVx - theirs.FovX) < 1e-9
        assert abs(ours.camera.FoVy - theirs.FovY) < 1e-9
        assert torch.allclose(ours.image, theirs.image, atol=1e-5)
    for ours, theirs in zip(scene.video_views, ref.video_cameras):
        assert np.allclose(ours.camera.R.numpy(), theirs.R, atol=1e-5)
        assert np.allclose(ours.camera.T.numpy(), theirs.T, atol=1e-5)
        # Official builds video times in float32 (torch.linspace), ours in
        # float64 (np.linspace): identical math, ~1e-8 representation noise.
        assert abs(ours.camera.time - float(theirs.time)) < 1e-6
    assert abs(scene.scene_extent - ref.nerf_normalization["radius"]) < 1e-4
    assert isinstance(scene.train_views[0], View)
    print("test_official_parity: passed (R/T/time/FoV/images/video/extent)")


if __name__ == "__main__":
    test_timeline_normalization()
    test_extrinsics_conversion_identity()
    test_scene_assembly()
    test_eval_false_merges_test_into_train()
    test_official_parity()
    print("test_dnerf.py: all 5 tests passed")

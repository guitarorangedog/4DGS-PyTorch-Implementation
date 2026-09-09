"""CPU + parity tests for data/dynerf.py. Run via `python3 -m tests.test_dynerf`."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from data import dynerf
from data.dynerf import (
    FRAME_COUNT,
    dynerf_split,
    llff_pose_to_R_T,
    read_poses_bounds,
    spiral_video_poses,
)
from data.scene import detect_dataset, load_scene


def _write_fixture(root: str, n_frames: int = 2) -> None:
    rng = np.random.default_rng(0)
    # Real poses_bounds.npy is [N_cams, 17]: flattened [3, 5] LLFF pose
    # (cols: R-ish | t | [H, W, focal] per row) + [near, far].
    rows = []
    for c in range(3):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        pose35 = np.concatenate(
            [Q, np.array([[0.5 * c], [0.1 * c], [0.0]]),
             np.array([[48.0], [64.0], [50.0]])], axis=1)  # last col = [H, W, focal]
        rows.append(np.concatenate([pose35.reshape(-1), [2.0, 6.0]]))
    np.save(os.path.join(root, "poses_bounds.npy"), np.stack(rows))
    for c in range(3):
        # Marker video files (official asserts cam-video count == pose count;
        # frames come from the pre-extracted images/ dirs, never decoded).
        open(os.path.join(root, f"cam{c:02d}.mp4"), "wb").close()
    for c in range(3):
        d = os.path.join(root, f"cam{c:02d}", "images")
        os.makedirs(d, exist_ok=True)
        for f in range(n_frames):
            arr = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr, "RGB").save(os.path.join(d, f"{f:04d}.png"))
    xyz = rng.random((50, 3)).astype(np.float32) * 2 - 1
    rgb = (rng.random((50, 3)) * 255).astype(np.float32)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"),
             ("nz", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")]
    el = np.empty(50, dtype=dtype)
    el[:] = list(map(tuple, np.concatenate(
        [xyz, np.zeros_like(xyz), rgb], axis=1)))
    PlyData([PlyElement.describe(el, "vertex")]).write(
        os.path.join(root, "points3D_downsample2.ply"))


def _official_dataset(root: str):
    sys.path.insert(0, "/tmp/opencode/4DGaussians")
    spec = importlib.util.spec_from_file_location(
        "official_ndc", "/tmp/opencode/4DGaussians/scene/neural_3D_dataset_NDC.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Neural3D_NDC_Dataset(
        root, "train", 1.0, time_scale=1,
        scene_bbox_min=[-2.5, -2.0, -1.0], scene_bbox_max=[2.5, 2.0, 1.0],
        eval_index=0)


def test_layout_split_time() -> None:
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        assert detect_dataset(root) == "dynerf"
        assert dynerf_split(3) == ([1, 2], [0])
        poses, focal, img_wh, near_fars = read_poses_bounds(root)
        assert img_wh == (1352, 1014) and focal == 25.0  # 50 / 2 quirk
        scene = load_scene(root, seed=0)
    assert scene.dataset_type == "dynerf" and scene.maxtime == 300
    # eval_index=0: train = cam01+cam02 frames, test = cam00 frames.
    assert [v.image_name for v in scene.train_views] == [
        "cam01_0000", "cam01_0001", "cam02_0000", "cam02_0001"]
    assert [v.image_name for v in scene.test_views] == ["cam00_0000", "cam00_0001"]
    assert [v.camera.time for v in scene.train_views] == [
        0.0, 1 / FRAME_COUNT, 0.0, 1 / FRAME_COUNT]
    assert scene.train_views[0].image.shape == (3, 1014, 1352)
    assert scene.train_views[0].camera.image_width == 1352
    assert scene.train_views[0].camera.image_height == 1014
    assert len(scene.video_views) == 300
    assert scene.video_views[0].image is None
    assert abs(scene.video_views[-1].camera.time - 299 / 300) < 1e-12
    assert scene.points.shape == (50, 3) and scene.colors.shape == (50, 3)
    assert scene.scene_extent > 0
    assert (scene.aabb_max > scene.aabb_min).all()


def test_official_parity() -> None:
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        ref = _official_dataset(root)
        scene = load_scene(root, seed=0)
        # Focal / resolution / frame times.
        assert ref.focal == [25.0, 25.0] and ref.img_wh == (1352, 1014)
        assert list(ref.image_times) == [0.0, 1 / 300, 0.0, 1 / 300]
        # Per-frame R/T conversion (official load_pose).
        for idx, view in enumerate(scene.train_views):
            R_ref, T_ref = ref.load_pose(idx)
            assert np.allclose(view.camera.R.numpy(), R_ref, atol=1e-5), idx
            assert np.allclose(view.camera.T.numpy(), T_ref, atol=1e-5), idx
        # Train image pixels (both LANCZOS-resize to (1352, 1014)).
        ref_img, _, ref_time = ref[0]
        assert ref_img.shape == scene.train_views[0].image.shape == (3, 1014, 1352)
        assert torch.allclose(scene.train_views[0].image, ref_img, atol=1e-6)
        assert ref_time == scene.train_views[0].camera.time == 0.0
        # FoV incl. the official format_infos H/W swap (FovX from H=1014).
        from gaussians.geometry import focal2fov
        assert abs(scene.train_views[0].camera.FoVx - focal2fov(25.0, 1014)) < 1e-9
        assert abs(scene.train_views[0].camera.FoVy - focal2fov(25.0, 1352)) < 1e-9
        # Spiral video trajectory vs official get_spiral.
        ours = spiral_video_poses(ref.poses_all, ref.near_fars, N_views=300)
        assert np.allclose(ours, ref.val_poses, atol=1e-8)
        print("test_official_parity: passed (R/T/times/images/FoV/spiral)")


def test_missing_pointcloud_errors() -> None:
    with tempfile.TemporaryDirectory() as root:
        _write_fixture(root)
        os.remove(os.path.join(root, "points3D_downsample2.ply"))
        try:
            load_scene(root, seed=0)
        except FileNotFoundError as e:
            assert "points3D_downsample2.ply" in str(e)
            return
    raise AssertionError("missing point cloud should raise")


if __name__ == "__main__":
    test_layout_split_time()
    test_official_parity()
    test_missing_pointcloud_errors()
    print("test_dynerf.py: all 3 tests passed")

"""REAL GPU integration test for training/render_static.py (Commit 10).

Trains a tiny static model, then: PLY reload -> test render -> video render
-> PNG/GT/mp4 outputs -> timestamp-independence probe.

Run via `python3 -m tests.test_render_static` (requires CUDA + extension).
"""

import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data.scene import load_scene
from training.render_static import (
    infer_sh_degree,
    load_static_model,
    render_splits,
    render_views,
    with_time,
)
from training.trainer_static import (
    StaticTrainerConfig,
    init_static_model,
    save_static_model,
    set_seed,
    train_static_scene,
)


def _write_scene(root: str, size: int = 64) -> None:
    def c2w(tx=0.0, ty=0.0):
        m = np.eye(4)
        m[:3, 3] = [tx, ty, 0.0]
        return m.tolist()

    train_frames = [
        {"file_path": f"train/r_{i}", "transform_matrix": c2w(0.1 * i), "time": t}
        for i, t in enumerate([0, 1])
    ]
    test_frames = [{"file_path": "test/r_0", "transform_matrix": c2w(0.0), "time": 1}]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    rng = np.random.default_rng(2)
    for rel in ("train/r_0", "train/r_1", "test/r_0"):
        arr = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        arr[..., 3] = 255
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        Image.fromarray(arr, "RGBA").save(os.path.join(root, rel + ".png"))


def test_render_static_pipeline() -> None:
    if not torch.cuda.is_available():
        print("test_render_static_pipeline: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = StaticTrainerConfig(iterations=3, seed=0, log_interval=3,
                                  max_sh_degree=0, densify_from_iter=100)
        model = init_static_model(scene, cfg, device=device)
        train_static_scene(scene, model, cfg, device=device)
        train_dir = os.path.join(root, "trained")
        paths = save_static_model(model, train_dir, cfg, [])
        del model
        torch.cuda.empty_cache()

        assert infer_sh_degree(paths["ply"]) == 0
        reloaded = load_static_model(paths["ply"], device=device)
        assert reloaded.num_points == 2000

        # Test / novel-view render.
        test_images = render_views(reloaded, scene.test_views, device=device)
        assert len(test_images) == 1
        img = test_images[0]
        assert tuple(img.shape) == (3, 64, 64)
        assert torch.isfinite(img).all() and img.min() >= 0.0 and img.max() <= 1.0

        # Video render (subset for speed) + full split save incl. GT + mp4.
        video_sub = scene.video_views[:8]
        frames = render_views(reloaded, video_sub, device=device)
        assert len(frames) == 8
        assert all(tuple(f.shape) == (3, 64, 64) and torch.isfinite(f).all()
                   for f in frames)

        out = render_splits(scene, reloaded, os.path.join(root, "renders"),
                            splits=("test", "video"), device=device)
        test_pngs = sorted(os.listdir(out["test"]["renders"]))
        assert test_pngs == ["00000_r_0.png"], test_pngs
        assert sorted(os.listdir(out["test"]["gt"])) == ["00000_r_0.png"]
        assert len(os.listdir(out["video"]["renders"])) == 160
        assert "gt" not in out["video"], "video views must not produce GT"
        assert os.path.exists(out["video"].get("mp4", "")), "mp4 missing"

        # Timestamp independence: identical pose/intrinsics, only time differs.
        view = scene.test_views[0]
        a = render_views(reloaded, [view], device=device)[0]
        b = render_views(reloaded, [with_time(view, 0.9)], device=device)[0]
        assert view.camera.time != 0.9
        assert torch.equal(a, b), \
            f"static render depends on time (max diff {(a - b).abs().max()})"
    print("test_render_static_pipeline: passed "
          f"(test mean={img.mean().item():.4f}, "
          f"video[0] mean={frames[0].mean().item():.4f}, time-independent)")


if __name__ == "__main__":
    test_render_static_pipeline()
    print("test_render_static.py: done")

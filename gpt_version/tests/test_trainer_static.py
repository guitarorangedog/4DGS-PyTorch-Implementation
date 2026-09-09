"""REAL GPU integration test for training/trainer_static.py.

Runs actual Scene loading -> model init -> render_view -> loss -> backward ->
Adam update -> densify/prune/reset on CUDA. No fakes, no CPU fallbacks.

Run via `python3 -m tests.test_trainer_static` (requires CUDA + extension).
"""

import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from data.scene import load_scene
from gaussians.gaussian_model import CanonicalGaussianModel
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
        {"file_path": f"train/r_{i}", "transform_matrix": c2w(0.1 * i, 0.0), "time": t}
        for i, t in enumerate([0, 1, 2])
    ]
    test_frames = [
        {"file_path": "test/r_0", "transform_matrix": c2w(0.0, 0.1), "time": 1}
    ]
    for name, frames in (("transforms_train.json", train_frames),
                         ("transforms_test.json", test_frames)):
        with open(os.path.join(root, name), "w") as f:
            json.dump({"camera_angle_x": 0.9, "frames": frames}, f)
    rng = np.random.default_rng(0)
    for rel in ("train/r_0", "train/r_1", "train/r_2", "test/r_0"):
        arr = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        arr[..., 3] = 255
        os.makedirs(os.path.join(root, os.path.dirname(rel)), exist_ok=True)
        Image.fromarray(arr, "RGBA").save(os.path.join(root, rel + ".png"))


def _smoke_cfg(**over) -> StaticTrainerConfig:
    # NOTE: opacity_reset_interval=100 keeps the official max_radii2D>20
    # prune gate inactive on this 64px toy scene (every toy Gaussian covers
    # >20px, so the faithful rule would prune the whole cloud — correct
    # behavior on real resolutions, degenerate here). reset_opacity itself
    # is unit-covered in Commit 5; densify + prune + full Adam path run here.
    kw = dict(iterations=6, seed=0, log_interval=1, max_sh_degree=0,
              densify_from_iter=2, densify_until_iter=6,
              densification_interval=2, densify_grad_threshold=0.0,
              pruning_interval=2, opacity_reset_interval=100, min_opacity=1e-9)
    kw.update(over)
    return StaticTrainerConfig(**kw)


def test_static_training_smoke() -> None:
    if not torch.cuda.is_available():
        print("test_static_training_smoke: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        set_seed(0)
        scene = load_scene(root, seed=0)
        cfg = _smoke_cfg()
        model = init_static_model(scene, cfg, device=device)
        n0 = model.num_points

        history = train_static_scene(scene, model, cfg, device=device)

        assert len(history) == 6
        for h in history:
            assert np.isfinite(h["loss"]) and np.isfinite(h["l1"])
            assert np.isfinite(h["xyz_grad_norm"]) and h["xyz_grad_norm"] > 0
        # Adam actually applied updates (shape-agnostic: densify changes N).
        assert model.optimizer.state[model._xyz]["exp_avg"].abs().sum() > 0
        assert history[-1]["loss"] != history[0]["loss"]
        assert model.num_points > n0, \
            f"densification path never fired ({model.num_points} vs {n0})"
        assert model.max_radii2D.max().item() > 0
        for g in model.optimizer.param_groups:  # optimizer still valid
            assert g["params"][0].shape[0] == model.num_points, g["name"]

        with tempfile.TemporaryDirectory() as out:
            paths = save_static_model(model, out, cfg, history)
            assert os.path.exists(paths["ply"]) and os.path.exists(paths["meta"])
            restored = CanonicalGaussianModel(max_sh_degree=0).load_ply(
                paths["ply"], device=device)
            assert torch.allclose(restored._xyz.detach(), model._xyz.detach(), atol=1e-6)
            assert restored.num_points == model.num_points
    print(f"test_static_training_smoke: passed "
          f"(loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}, "
          f"n {n0} -> {model.num_points})")


def test_deterministic_sampling() -> None:
    if not torch.cuda.is_available():
        print("test_deterministic_sampling: skipped (no CUDA)")
        return
    device = torch.device("cuda")
    losses = []
    with tempfile.TemporaryDirectory() as root:
        _write_scene(root)
        for _ in range(2):
            set_seed(0)
            scene = load_scene(root, seed=0)
            model = init_static_model(scene, _smoke_cfg(), device=device)
            hist = train_static_scene(scene, model, _smoke_cfg(), device=device)
            losses.append([h["loss"] for h in hist])
            del model
            torch.cuda.empty_cache()
    assert torch.allclose(torch.tensor(losses[0]), torch.tensor(losses[1]),
                          atol=1e-6), losses
    print("test_deterministic_sampling: passed")


if __name__ == "__main__":
    test_static_training_smoke()
    test_deterministic_sampling()
    print("test_trainer_static.py: done")

"""Static 3D-GS trainer: the trainable baseline (Commit 9) [STATIC].

Connects Commits 1-8 into a real training pipeline on the faithful CUDA
rasterizer. Deliberately STATIC: ``Camera.time`` exists on every view but is
never read here — Commit 14 will show the exact diff where canonical
rendering becomes time-conditioned 4D rendering.

Per-iteration order (mirrors official ``train.py`` static path):

1. ``update_xyz_lr`` (official ``update_learning_rate``).
2. SH degree step every 1000 iters (official ``oneupSHdegree``).
3. Sample one train view uniformly (``random.Random(seed)`` stream).
4. ``render_view`` canonical Gaussians (+ white/black background).
5. ``reconstruction_loss`` -> ``backward`` (NaN raises; official re-execs).
6. ``max_radii2D`` update + ``add_densification_stats`` from
   ``viewspace_points.grad`` (requires the ``retain_grad`` in ``render_view``).
7. Under ``torch.no_grad``: densify / prune / opacity-reset at their
   intervals (BEFORE the optimizer step, as in official code).
8. ``optimizer.step()`` unless this is the final iteration, then
   ``zero_grad`` (official ``if iteration < opt.iterations``).
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from data.scene import Scene
from gaussians.densify import (
    add_densification_stats,
    densify,
    prune_by_opacity_and_size,
    reset_opacity,
)
from gaussians.gaussian_model import CanonicalGaussianModel
from gaussians.rasterizer import render_view
from training.losses import l1_loss, reconstruction_loss, ssim
from training.optim import StaticTrainingConfig, setup_static_optimizer, update_xyz_lr

__all__ = [
    "StaticTrainerConfig",
    "set_seed",
    "init_static_model",
    "train_static_scene",
    "save_static_model",
]


@dataclass
class StaticTrainerConfig:
    """Static-training hyperparameters (official 3D-GS / 4DGS conventions)."""

    iterations: int = 3000
    optim: StaticTrainingConfig = field(default_factory=StaticTrainingConfig)
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    pruning_interval: int = 100
    opacity_reset_interval: int = 3000
    min_opacity: float = 0.005
    max_screen_size: float = 20.0
    log_interval: int = 10
    seed: int = 0
    max_sh_degree: int = 3
    background: tuple = (1.0, 1.0, 1.0)  # white; official D-NeRF default


def set_seed(seed: int) -> None:
    """Seed Python / NumPy / Torch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_static_model(scene: Scene, cfg: StaticTrainerConfig,
                      device: torch.device | str = "cuda") -> CanonicalGaussianModel:
    """Build the canonical model from ``Scene`` points/colors.

    Sets ``spatial_lr_scale`` from ``scene.scene_extent`` (official
    ``cameras_extent``) and installs the Commit 8 optimizer + densify stats.
    """
    device = torch.device(device)
    model = CanonicalGaussianModel(max_sh_degree=cfg.max_sh_degree).to(device)
    model.create_from_pointcloud(scene.points, scene.colors, device=device)
    setup_static_optimizer(model, cfg.optim, spatial_lr_scale=scene.scene_extent)
    return model


def _should_densify(cfg: StaticTrainerConfig, iteration: int, n: int) -> bool:
    return (
        iteration > cfg.densify_from_iter
        and iteration < cfg.densify_until_iter
        and iteration % cfg.densification_interval == 0
    )


def train_static_scene(scene: Scene, model: CanonicalGaussianModel,
                       cfg: StaticTrainerConfig,
                       device: torch.device | str = "cuda") -> list[dict]:
    """Run the static training loop; returns per-log-step history dicts.

    Each history entry: ``iteration, loss, l1, ssim, n_points, xyz_lr,
    xyz_grad_norm, view`` (view name sampled that step).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("static training requires CUDA + the rasterizer extension.")
    device = torch.device(device)
    bg = torch.tensor(cfg.background, dtype=torch.float32, device=device)
    rng = random.Random(cfg.seed)
    history: list[dict] = []

    for iteration in range(1, cfg.iterations + 1):
        xyz_lr = update_xyz_lr(model, iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()

        view = scene.train_views[rng.randrange(len(scene.train_views))]
        pkg = render_view(view.camera, model, bg_color=bg, device=device)
        gt = view.image.to(device)
        pred = pkg["render"]
        l1 = l1_loss(pred.unsqueeze(0), gt.unsqueeze(0))
        s = ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        loss = reconstruction_loss(pred.unsqueeze(0), gt.unsqueeze(0),
                                   cfg.optim.lambda_dssim)
        if torch.isnan(loss):
            raise RuntimeError(f"loss is NaN at iteration {iteration}")
        loss.backward()

        with torch.no_grad():
            vis = pkg["visibility_filter"]
            radii = pkg["radii"]
            model.max_radii2D[vis] = torch.max(model.max_radii2D[vis], radii[vis])
            add_densification_stats(model, pkg["viewspace_points"].grad, vis)
            grad_norm = float(model._xyz.grad.norm().item()) if model._xyz.grad is not None else 0.0

            if _should_densify(cfg, iteration, model.num_points):
                densify(model, cfg.densify_grad_threshold, scene.scene_extent,
                        model.percent_dense)
            if (iteration > cfg.densify_from_iter
                    and iteration % cfg.pruning_interval == 0):
                size_thr = cfg.max_screen_size if iteration > cfg.opacity_reset_interval else None
                prune_by_opacity_and_size(model, cfg.min_opacity,
                                          scene.scene_extent, size_thr)
            if (iteration < cfg.densify_until_iter
                    and iteration % cfg.opacity_reset_interval == 0):
                reset_opacity(model)

            if iteration % cfg.log_interval == 0 or iteration == cfg.iterations:
                history.append({
                    "iteration": iteration,
                    "loss": float(loss.item()),
                    "l1": float(l1.item()),
                    "ssim": float(s.item()),
                    "n_points": model.num_points,
                    "xyz_lr": xyz_lr,
                    "xyz_grad_norm": grad_norm,
                    "view": view.image_name,
                })
                print(f"[iter {iteration:6d}] loss={loss.item():.6f} "
                      f"l1={l1.item():.6f} ssim={s.item():.4f} "
                      f"n={model.num_points} lr={xyz_lr:.2e} view={view.image_name}",
                      flush=True)

        if iteration < cfg.iterations:
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

    return history


def save_static_model(model: CanonicalGaussianModel, output_dir: str,
                      cfg: StaticTrainerConfig, history: list[dict]) -> dict[str, str]:
    """Save canonical PLY + minimal training metadata. Returns paths."""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    meta_path = os.path.join(output_dir, "train_meta.json")
    model.save_ply(ply_path)
    meta = {
        "iterations": cfg.iterations,
        "seed": cfg.seed,
        "lambda_dssim": cfg.optim.lambda_dssim,
        "n_points": model.num_points,
        "active_sh_degree": model.active_sh_degree,
        "final_loss": history[-1]["loss"] if history else None,
        "static": True,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"ply": ply_path, "meta": meta_path}

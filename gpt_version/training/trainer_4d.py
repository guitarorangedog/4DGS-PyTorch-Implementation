"""Coarse-to-fine 4D-GS trainer: the first end-to-end 4DGS pipeline (Commit 16).

Official schedule reproduced (``train.py::training`` + ``OptimizationParams``):

- COARSE (iters ``1..coarse_iterations``, default 3000): static path only.
  The renderer bypasses deformation (official ``"coarse" in stage``); the
  full 8-group optimizer exists but deformation/grid params receive no
  gradients and stay frozen in effect. Coarse densify thresholds.
- FINE (iters ``1..fine_iterations``, default 20000 — INDEPENDENT counter):
  ``camera.time -> DeformationField -> CUDA render -> L1 + λ(1-SSIM) + TV``.
  Fresh optimizer + zeroed stats (official ``training_setup`` per stage);
  SH degree continues (model persists, capped at max). Fine thresholds
  interpolate ``init -> after`` over ``densify_until_iter``.

Per-iteration order (official ``scene_reconstruction``):

1. LR updates (xyz + deformation + grid).
2. SH step every 1000.
3. Sample one train view (seeded RNG).
4. Render (static coarse / deformed fine).
5. Loss -> backward (NaN raises; official re-execs).
6. If ``iteration < densify_until``: max_radii2D + stats, then
   densify / prune / opacity-reset at intervals (BEFORE the step).
7. Step unless ``stage == fine and iteration == fine_iterations``
   (official ``iteration < opt.iterations`` bites only at fine's last iter).
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from data.scene import Scene, load_scene
from deformation.field import DeformationField, FieldConfig
from deformation.hexplane import HexPlaneConfig
from deformation.render_4d import render_deformed_view
from deformation.regularize import RegWeights, regularization_terms
from gaussians.densify import (
    add_densification_stats,
    densify,
    prune_by_opacity_and_size,
    reset_opacity,
)
from gaussians.gaussian_model import CanonicalGaussianModel
from gaussians.rasterizer import render_view
from training.losses import l1_loss, reconstruction_loss_4dgs, ssim
from training.optim import (
    DeformationOptimConfig,
    StaticTrainingConfig,
    setup_4d_optimizer,
    update_4d_lrs,
)
from training.trainer_static import set_seed

__all__ = [
    "FourDTrainerConfig",
    "init_4d_model",
    "train_coarse",
    "train_fine",
    "train_4d",
    "save_4d_model",
    "fine_threshold",
]

#: Official densify-count guard (densify skipped above this N).
MAX_DENSIFY_POINTS = 360_000


#: Official D-NeRF HexPlane config (``arguments/dnerf/dnerf_default.py``:
#: ``multires = [1, 2]`` — narrower than the global ``[1, 2, 4, 8]`` default,
#: which ``HexPlaneConfig`` itself retains). The Commit 16 trainer targets
#: D-NeRF, so its default field uses this.
def _dnerf_field_config() -> FieldConfig:
    hexplane = HexPlaneConfig()
    hexplane.multires = [1, 2]
    return FieldConfig(hexplane=hexplane)


@dataclass
class FourDTrainerConfig:
    """Coarse-to-fine hyperparameters (official D-NeRF defaults)."""

    coarse_iterations: int = 3000
    fine_iterations: int = 20_000
    optim: StaticTrainingConfig = field(default_factory=StaticTrainingConfig)
    deform_optim: DeformationOptimConfig = field(default_factory=DeformationOptimConfig)
    reg: RegWeights = field(default_factory=RegWeights)
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    densify_grad_threshold_coarse: float = 0.0002
    densify_grad_threshold_fine_init: float = 0.0002
    densify_grad_threshold_fine_after: float = 0.0002
    # Official D-NeRF override (dnerf_default.py): pruning every 8000 iters,
    # NOT the global default of 100.
    pruning_interval: int = 8000
    opacity_reset_interval: int = 3000
    opacity_threshold_coarse: float = 0.005
    opacity_threshold_fine_init: float = 0.005
    opacity_threshold_fine_after: float = 0.005
    max_screen_size: float = 20.0
    log_interval: int = 10
    seed: int = 0
    max_sh_degree: int = 3
    field: FieldConfig = field(default_factory=_dnerf_field_config)
    background: tuple = (1.0, 1.0, 1.0)


def fine_threshold(init: float, after: float, iteration: int, until_iter: int) -> float:
    """Official fine-stage linear interpolation ``init -> after`` over ``until``.

    With official defaults (``init == after``) this is constant; the formula
    is preserved for config generality.
    """
    return init - iteration * (init - after) / until_iter


def _should_prune(cfg: FourDTrainerConfig, iteration: int) -> bool:
    """Official prune cadence (``pruning_interval``, D-NeRF: 8000)."""
    return (iteration > cfg.densify_from_iter
            and iteration % cfg.pruning_interval == 0)


def init_4d_model(scene: Scene, cfg: FourDTrainerConfig,
                  device: torch.device | str = "cuda"
                  ) -> tuple[CanonicalGaussianModel, DeformationField]:
    """Build canonical model + field; AABB from scene BEFORE device move."""
    device = torch.device(device)
    model = CanonicalGaussianModel(max_sh_degree=cfg.max_sh_degree).to(device)
    model.create_from_pointcloud(scene.points, scene.colors, device=device)
    field = DeformationField(cfg.field)
    field.set_aabb(scene.aabb_max, scene.aabb_min)  # official (max, min) order
    field = field.to(device)
    setup_4d_optimizer(model, field, cfg.optim, cfg.deform_optim,
                       spatial_lr_scale=scene.scene_extent)
    return model, field


def _stage_loop(scene: Scene, model: CanonicalGaussianModel, field: DeformationField,
                cfg: FourDTrainerConfig, stage: str, iterations: int,
                device: torch.device, start_iteration: int = 1, loop_rng=None,
                checkpoint_cb=None) -> list[dict]:
    """One stage loop. ``start_iteration`` resumes mid-stage (post-checkpoint
    state represents end-of-iteration ``start_iteration - 1``); ``loop_rng``
    continues the camera stream; ``checkpoint_cb(stage, iteration, model,
    field, loop_rng)`` fires end-of-iteration (post-step) when provided."""
    assert stage in ("coarse", "fine")
    bg = torch.tensor(cfg.background, dtype=torch.float32, device=device)
    if loop_rng is None:
        loop_rng = random.Random(cfg.seed + (0 if stage == "coarse" else 10_000))
    rng = loop_rng
    history: list[dict] = []

    for iteration in range(start_iteration, iterations + 1):
        lrs = update_4d_lrs(model, iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()

        view = scene.train_views[rng.randrange(len(scene.train_views))]
        if stage == "coarse":
            pkg = render_view(view.camera, model, bg_color=bg, device=device)
        else:
            pkg = render_deformed_view(view.camera, model, field,
                                       bg_color=bg, device=device)
        gt = view.image.to(device)
        pred = pkg["render"].unsqueeze(0)
        target = gt.unsqueeze(0)
        l1 = l1_loss(pred, target)
        s = ssim(pred, target)
        loss = reconstruction_loss_4dgs(pred, target, cfg.optim.lambda_dssim)
        reg_total = None
        if stage == "fine" and cfg.reg.time_smoothness != 0:
            terms = regularization_terms(field)
            reg_total = (cfg.reg.plane_tv * terms.spatial
                         + cfg.reg.time_smoothness * terms.temporal
                         + cfg.reg.l1_time_planes * terms.l1_time)
            loss = loss + reg_total
        if torch.isnan(loss):
            raise RuntimeError(f"loss is NaN at {stage} iteration {iteration}")
        loss.backward()

        with torch.no_grad():
            if iteration < cfg.densify_until_iter:
                vis = pkg["visibility_filter"]
                radii = pkg["radii"]
                model.max_radii2D[vis] = torch.max(model.max_radii2D[vis], radii[vis])
                add_densification_stats(model, pkg["viewspace_points"].grad, vis)
                if stage == "coarse":
                    dens_thr = cfg.densify_grad_threshold_coarse
                    op_thr = cfg.opacity_threshold_coarse
                else:
                    dens_thr = fine_threshold(cfg.densify_grad_threshold_fine_init,
                                              cfg.densify_grad_threshold_fine_after,
                                              iteration, cfg.densify_until_iter)
                    op_thr = fine_threshold(cfg.opacity_threshold_fine_init,
                                            cfg.opacity_threshold_fine_after,
                                            iteration, cfg.densify_until_iter)
                if (iteration > cfg.densify_from_iter
                        and iteration % cfg.densification_interval == 0
                        and model.num_points < MAX_DENSIFY_POINTS):
                    densify(model, dens_thr, scene.scene_extent, model.percent_dense)
                if _should_prune(cfg, iteration):
                    size_thr = cfg.max_screen_size if iteration > cfg.opacity_reset_interval else None
                    prune_by_opacity_and_size(model, op_thr, scene.scene_extent, size_thr)
                if iteration % cfg.opacity_reset_interval == 0:
                    reset_opacity(model)
            grad_norm = float(model._xyz.grad.norm().item()) if model._xyz.grad is not None else 0.0

            if iteration % cfg.log_interval == 0 or iteration == iterations:
                entry = {"stage": stage, "iteration": iteration,
                         "loss": float(loss.item()), "l1": float(l1.item()),
                         "ssim": float(s.item()), "n_points": model.num_points,
                         "xyz_lr": lrs["xyz"], "xyz_grad_norm": grad_norm,
                         "view": view.image_name}
                if reg_total is not None:
                    entry["reg"] = float(reg_total.item())
                history.append(entry)
                print(f"[{stage} {iteration:6d}] loss={loss.item():.6f} "
                      f"l1={l1.item():.6f} ssim={s.item():.4f} "
                      + (f"reg={reg_total.item():.2e} " if reg_total is not None else "")
                      + f"n={model.num_points} view={view.image_name}", flush=True)

        if not (stage == "fine" and iteration == iterations):
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

        if checkpoint_cb is not None:
            checkpoint_cb(stage, iteration, model, field, rng)

    return history


def train_coarse(scene: Scene, model: CanonicalGaussianModel, field: DeformationField,
                 cfg: FourDTrainerConfig, device: torch.device | str = "cuda",
                 start_iteration: int = 1, loop_rng=None, checkpoint_cb=None) -> list[dict]:
    """Coarse stage: fresh optimizer, static rendering (official semantics)."""
    setup_4d_optimizer(model, field, cfg.optim, cfg.deform_optim,
                       spatial_lr_scale=scene.scene_extent)
    return _stage_loop(scene, model, field, cfg, "coarse", cfg.coarse_iterations,
                       torch.device(device), start_iteration, loop_rng, checkpoint_cb)


def train_fine(scene: Scene, model: CanonicalGaussianModel, field: DeformationField,
               cfg: FourDTrainerConfig, device: torch.device | str = "cuda",
               start_iteration: int = 1, loop_rng=None, checkpoint_cb=None) -> list[dict]:
    """Fine stage: fresh optimizer, time-conditioned rendering + TV regs."""
    setup_4d_optimizer(model, field, cfg.optim, cfg.deform_optim,
                       spatial_lr_scale=scene.scene_extent)
    return _stage_loop(scene, model, field, cfg, "fine", cfg.fine_iterations,
                       torch.device(device), start_iteration, loop_rng, checkpoint_cb)


def _checkpoint_callback(output_dir: str, interval: int, cfg: FourDTrainerConfig):
    """Build an end-of-iteration checkpoint callback (None when interval<=0)."""
    if interval <= 0:
        return None
    from training.checkpoints import checkpoint_filename, save_checkpoint

    def cb(stage, iteration, model, field, loop_rng):
        if iteration % interval == 0:
            save_checkpoint(
                checkpoint_filename(output_dir, stage, iteration),
                stage=stage, iteration=iteration, model=model, field=field,
                loop_rng=loop_rng,
                extra_meta={"seed": cfg.seed,
                            "coarse_iterations": cfg.coarse_iterations,
                            "fine_iterations": cfg.fine_iterations})
    return cb


def train_4d(scene: Scene, model: CanonicalGaussianModel, field: DeformationField,
             cfg: FourDTrainerConfig, device: torch.device | str = "cuda",
             resume: str | None = None, output_dir: str | None = None,
             checkpoint_interval: int = 0) -> dict[str, list[dict]]:
    """Full coarse-to-fine run (or resume). Returns ``{"coarse": [...], "fine": [...]}``.

    Args:
        resume: checkpoint path. Coarse checkpoints continue coarse at
            ``N + 1`` then run fine fresh; fine checkpoints continue fine at
            ``N + 1`` (coarse already complete — history ``"coarse"`` is empty).
        output_dir: required when ``checkpoint_interval > 0`` (or resume
            checkpoints should also be written).
        checkpoint_interval: save end-of-iteration checkpoints every N iters
            (0 = off).
    """
    from training.checkpoints import load_checkpoint, restore_4d_state

    device = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("4D training requires CUDA + the rasterizer extension.")
    cb = _checkpoint_callback(output_dir, checkpoint_interval, cfg) \
        if output_dir is not None else None
    if resume is None:
        coarse_hist = train_coarse(scene, model, field, cfg, device,
                                   checkpoint_cb=cb)
        fine_hist = train_fine(scene, model, field, cfg, device,
                               checkpoint_cb=cb)
        return {"coarse": coarse_hist, "fine": fine_hist}
    payload = load_checkpoint(resume, map_location=device)
    loop_rng = random.Random()
    restore_4d_state(model, field, payload, cfg.optim, cfg.deform_optim,
                     device, loop_rng)
    if payload["stage"] == "coarse":
        coarse_hist = _stage_loop(scene, model, field, cfg, "coarse",
                                  cfg.coarse_iterations, device,
                                  payload["iteration"] + 1, loop_rng, cb)
        fine_hist = train_fine(scene, model, field, cfg, device,
                               checkpoint_cb=cb)
    else:
        coarse_hist = []
        fine_hist = _stage_loop(scene, model, field, cfg, "fine",
                                cfg.fine_iterations, device,
                                payload["iteration"] + 1, loop_rng, cb)
    return {"coarse": coarse_hist, "fine": fine_hist}


def save_4d_model(model: CanonicalGaussianModel, field: DeformationField,
                  output_dir: str, cfg: FourDTrainerConfig,
                  history: dict[str, list[dict]]) -> dict[str, str]:
    """Save canonical PLY + deformation weights + metadata (pre-checkpoint)."""
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    deform_path = os.path.join(output_dir, "deformation.pth")
    meta_path = os.path.join(output_dir, "train_meta.json")
    model.save_ply(ply_path)
    torch.save(field.state_dict(), deform_path)
    meta = {
        "coarse_iterations": cfg.coarse_iterations,
        "fine_iterations": cfg.fine_iterations,
        "seed": cfg.seed,
        "lambda_dssim": cfg.optim.lambda_dssim,
        "n_points": model.num_points,
        "active_sh_degree": model.active_sh_degree,
        "final_coarse_loss": history["coarse"][-1]["loss"] if history["coarse"] else None,
        "final_fine_loss": history["fine"][-1]["loss"] if history["fine"] else None,
        "static": False,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"ply": ply_path, "deformation": deform_path, "meta": meta_path}

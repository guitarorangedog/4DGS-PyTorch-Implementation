"""Training checkpoints: save / restore / resume for 4DGS [INFRA].

Official behavior traced (``train.py`` + ``GaussianModel.capture/restore``):

- ``capture()`` saves raw Gaussian tensors (``_xyz``, ``_scaling``,
  ``_rotation``, ``_opacity``, ``_features_dc/rest``), ``active_sh_degree``,
  deformation ``state_dict``, ``max_radii2D``, ``xyz_gradient_accum``,
  ``denom``, ``optimizer.state_dict()`` and ``spatial_lr_scale`` — as a
  positional tuple plus iteration, ``chkpnt_{stage}_{iter}.pth``.
- ``restore()`` reloads deformation weights, calls ``training_setup`` (FRESH
  optimizer with identical group ORDER), overwrites accumulators, then
  ``optimizer.load_state_dict`` (moments keyed by group order — shapes must
  match the restored topology).
- Checkpoints are end-of-iteration state (saved AFTER ``optimizer.step``);
  resume continues at ``N + 1``. Stage is matched by checkpoint PATH name.
- Official saves NO RNG state.

Our format (single file, explicit dict — no positional tuple)::

    checkpoint_{stage}_{iteration:06d}.pth  # iteration = last COMPLETED iter

Contents: raw Gaussian state, SH degree, densify stats, field ``state_dict``
(+ config for architecture check), full Adam ``state_dict`` (all eight
groups incl. ``exp_avg``/``exp_avg_sq``/``step``), stage + iteration,
optimizer/schedule metadata, and RNG state (stage-loop stream + torch CPU +
torch CUDA). Rendering-only artifacts (``point_cloud.ply``,
``deformation.pth``) stay separate and sufficient for inference.
"""

import os

import torch

__all__ = [
    "checkpoint_filename",
    "save_checkpoint",
    "load_checkpoint",
    "restore_4d_state",
]


def checkpoint_filename(output_dir: str, stage: str, iteration: int) -> str:
    """``<dir>/checkpoint_{stage}_{iteration:06d}.pth``."""
    return os.path.join(output_dir, f"checkpoint_{stage}_{iteration:06d}.pth")


def _rng_snapshot(loop_rng) -> dict:
    return {
        "loop": loop_rng.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _rng_restore(snap: dict, loop_rng) -> None:
    # Setter quirk (verified on torch 2.8): torch.set_rng_state AND
    # torch.cuda.set_rng_state_all both require CPU ByteTensors, regardless
    # of the checkpoint's map_location. Coerce everything to CPU first.
    loop_rng.setstate(snap["loop"])
    torch_state = snap["torch"]
    torch.set_rng_state(torch_state if torch_state.device.type == "cpu"
                        else torch_state.cpu())
    cuda_states = snap["cuda"]
    if isinstance(cuda_states, torch.Tensor):
        cuda_states = [cuda_states]
    torch.cuda.set_rng_state_all([
        s if s.device.type == "cpu" else s.cpu() for s in cuda_states
    ])


def save_checkpoint(path: str, *, stage: str, iteration: int, model, field,
                    loop_rng, extra_meta: dict | None = None) -> str:
    """Save end-of-iteration resume state (call AFTER ``optimizer.step``).

    Args:
        path: destination ``.pth`` file.
        stage: ``"coarse"`` or ``"fine"``.
        iteration: last completed iteration (resume continues at ``+ 1``).
        model: canonical model (raw params + stats + optimizer).
        field: deformation field (weights + config recorded).
        loop_rng: the stage loop's ``random.Random`` (camera stream).
        extra_meta: small run metadata (stored verbatim for sanity checks).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "stage": stage,
        "iteration": int(iteration),
        "model": {
            "xyz": model._xyz.detach().cpu(),
            "scaling": model._scaling.detach().cpu(),
            "rotation": model._rotation.detach().cpu(),
            "opacity": model._opacity.detach().cpu(),
            "features_dc": model._features_dc.detach().cpu(),
            "features_rest": model._features_rest.detach().cpu(),
            "active_sh_degree": int(model.active_sh_degree),
            "max_sh_degree": int(model.max_sh_degree),
            "spatial_lr_scale": float(model.spatial_lr_scale),
            "percent_dense": float(model.percent_dense),
        },
        "stats": {
            "xyz_gradient_accum": model.xyz_gradient_accum.detach().cpu(),
            "denom": model.denom.detach().cpu(),
            "max_radii2D": model.max_radii2D.detach().cpu(),
        },
        "field": field.state_dict(),
        "field_config": {
            "multires": list(field.hexplane.config.multires),
            "output_coordinate_dim": field.hexplane.config.output_coordinate_dim,
            "resolution": list(field.hexplane.config.resolution),
            "width": field.decoder.config.width,
            "depth": field.decoder.config.depth,
            "apply_rotation": field.config.apply_rotation,
            "enable_aux": field.config.enable_aux,
        },
        "optimizer": model.optimizer.state_dict(),
        "rng": _rng_snapshot(loop_rng),
        "meta": dict(extra_meta or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Load a checkpoint payload (tensors on ``map_location``)."""
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_4d_model_params(model, payload: dict, device) -> None:
    """Rebuild raw canonical params + stats on ``model`` (in place)."""
    from torch import nn

    m = payload["model"]
    assert m["max_sh_degree"] == model.max_sh_degree, "SH degree mismatch"
    model._xyz = nn.Parameter(m["xyz"].to(device).requires_grad_(True))
    model._scaling = nn.Parameter(m["scaling"].to(device).requires_grad_(True))
    model._rotation = nn.Parameter(m["rotation"].to(device).requires_grad_(True))
    model._opacity = nn.Parameter(m["opacity"].to(device).requires_grad_(True))
    model._features_dc = nn.Parameter(m["features_dc"].to(device).requires_grad_(True))
    model._features_rest = nn.Parameter(m["features_rest"].to(device).requires_grad_(True))
    model.active_sh_degree = int(m["active_sh_degree"])
    model.spatial_lr_scale = float(m["spatial_lr_scale"])
    model.percent_dense = float(m["percent_dense"])
    s = payload["stats"]
    model.xyz_gradient_accum = s["xyz_gradient_accum"].to(device)
    model.denom = s["denom"].to(device)
    model.max_radii2D = s["max_radii2D"].to(device)


def restore_4d_state(model, field, payload: dict, static_cfg, deform_cfg, device,
                     loop_rng=None):
    """Full resume reconstruction (in place). Steps:

    1. Check field architecture matches the checkpoint config.
    2. Rebuild raw model params + stats; reload field weights.
    3. Fresh 8-group optimizer (official ``training_setup`` semantics) via
       :func:`setup_4d_optimizer`, then ``optimizer.load_state_dict`` to
       restore moments/counters (group ORDER is the contract).
    4. Restore RNG streams (loop stream returned via ``loop_rng`` when given).

    Returns:
        The restored loop RNG state (also installed into ``loop_rng``).
    """
    from training.optim import setup_4d_optimizer

    fc = payload["field_config"]
    ours = field.hexplane.config
    assert list(fc["multires"]) == list(ours.multires), "HexPlane multires mismatch"
    assert fc["output_coordinate_dim"] == ours.output_coordinate_dim
    assert fc["width"] == field.decoder.config.width and fc["depth"] == field.decoder.config.depth
    assert fc["apply_rotation"] == field.config.apply_rotation
    assert fc["enable_aux"] == (field.aux is not None)

    restore_4d_model_params(model, payload, device)
    field.load_state_dict({k: v.to(device) for k, v in payload["field"].items()})
    setup_4d_optimizer(model, field, static_cfg, deform_cfg,
                       spatial_lr_scale=model.spatial_lr_scale)
    # setup() zeroes densify stats via init_densify_stats: re-apply saved ones.
    s = payload["stats"]
    model.xyz_gradient_accum = s["xyz_gradient_accum"].to(device)
    model.denom = s["denom"].to(device)
    model.max_radii2D = s["max_radii2D"].to(device)
    model.optimizer.load_state_dict(payload["optimizer"])
    # Moments restore onto whatever device the payload holds (checkpoint may
    # have been mapped to CPU); migrate them to the training device so the
    # next step sees matching param/state devices.
    for state in model.optimizer.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in state and state[k].device != torch.device(device):
                state[k] = state[k].to(device)
    model.optimizer.zero_grad(set_to_none=True)
    if loop_rng is not None:
        _rng_restore(payload["rng"], loop_rng)
    return payload["rng"]

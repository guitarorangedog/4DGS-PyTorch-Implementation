"""Minimal 4D training step (Commit 14 integration, not the full trainer).

Conceptually::

    camera -> deform at camera.time -> CUDA render -> loss -> backward

Proves image-space supervision reaches canonical parameters, decoder
weights and HexPlane grids. No densification, no regularization, no
coarse/fine schedule (Commit 16); no optimizer construction here — the
caller owns stepping (see ``tests/test_4d_integration.py``).
"""

import torch

from deformation.render_4d import render_deformed_view
from training.losses import l1_loss, reconstruction_loss, ssim

__all__ = ["train_step_4d"]


def train_step_4d(camera, gt_image: torch.Tensor, model, field,
                  bg_color=(1.0, 1.0, 1.0), lambda_dssim: float = 0.0,
                  scaling_modifier: float = 1.0,
                  device: torch.device | str = "cuda") -> dict:
    """One deform-render-loss-backward step.

    Args:
        camera: view carrying the conditioning ``camera.time``.
        gt_image: ``[3, H, W]`` ground truth in ``[0, 1]`` (any device).
        model: canonical Gaussians; field: deformation field.
        bg_color / lambda_dssim / scaling_modifier / device: as usual.

    Returns:
        ``{loss, l1, ssim, pkg, state}`` with ``loss`` already backpropagated.
    """
    device = torch.device(device)
    gt = gt_image.to(device)
    pkg = render_deformed_view(camera, model, field, bg_color=bg_color,
                               scaling_modifier=scaling_modifier, device=device)
    pred = pkg["render"].unsqueeze(0)
    target = gt.unsqueeze(0)
    l1 = l1_loss(pred, target)
    s = ssim(pred, target)
    loss = reconstruction_loss(pred, target, lambda_dssim)
    if torch.isnan(loss):
        raise RuntimeError("4D training-step loss is NaN")
    loss.backward()
    return {"loss": loss, "l1": l1, "ssim": s, "pkg": pkg, "state": pkg["state"]}

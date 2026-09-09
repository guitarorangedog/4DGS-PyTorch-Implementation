"""Image-quality metrics with official 4DGS semantics [INFRA].

Conventions (traced from ``metrics.py`` + ``utils/image_utils.py``):

- Tensors ``[B, 3, H, W]``, float32, range ``[0, 1]`` (PIL ``to_tensor`` +
  ``[:, :3]`` — alpha dropped, never compared as RGBA).
- Per-image values first, then ARITHMETIC MEAN over images (official
  ``torch.tensor(list).mean()``).
- PSNR: per-image MSE over all pixels/channels,
  ``20 * log10(1 / sqrt(mse))`` (range 1.0); identical images -> ``inf``.
- SSIM: reuses :func:`training.losses.ssim` — the SAME function as official
  ``utils/loss_utils.ssim`` (Commit 8 parity: 0.0 diff), not a second
  implementation.
- LPIPS: exact vendored official fork (``evaluation/_lpips``), backbones
  ``alex`` + ``vgg`` (official ``metrics.py`` reports both), inputs ``[0,1]``
  (the fork z-scores internally; NO ``[-1, 1]`` conversion — the PyPI
  ``lpips`` package differs in convention and is NOT used).
"""

import torch

from evaluation._lpips import lpips as _lpips
from training.losses import ssim

__all__ = ["psnr", "ssim", "lpips_value", "evaluate_pair"]


@torch.no_grad()
def psnr(img1: torch.Tensor, img2: torch.Tensor, mask=None) -> torch.Tensor:
    """Per-image PSNR, ``[B, 1]`` (official ``image_utils.psnr``, incl. mask path)."""
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)
        mask = mask.flatten(1).repeat(3, 1)
        mask = torch.where(mask != 0, True, False)
        img1 = img1[mask]
        img2 = img2[mask]
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    value = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None and torch.isinf(value).any():
        value = value[~torch.isinf(value)]
    return value


@torch.no_grad()
def lpips_value(img1: torch.Tensor, img2: torch.Tensor, net_type: str = "alex",
                model=None) -> torch.Tensor:
    """LPIPS distance for ``[0, 1]`` inputs (official fork, vgg/alex).

    Args:
        img1, img2: ``[B, 3, H, W]`` in ``[0, 1]`` (no rescaling).
        net_type: ``"alex"`` or ``"vgg"``.
        model: cached ``LPIPS`` module (built on demand; downloads weights
            once via ``torch.hub``).

    Returns:
        ``[B, 1, 1, 1]`` distances (mirrors official return shape).
    """
    assert net_type in ("alex", "vgg"), net_type
    if model is None:
        from evaluation._lpips.modules.lpips import LPIPS
        model = LPIPS(net_type, "0.1").to(img1.device)
        model.eval()
    return model(img1, img2)


@torch.no_grad()
def evaluate_pair(pred: torch.Tensor, gt: torch.Tensor, lpips_models=None) -> dict:
    """All metrics for one ``[1, 3, H, W]`` pair (floats, ``inf``-preserving)."""
    if lpips_models is None:
        lpips_models = {}
    out = {
        "psnr": float(psnr(pred, gt).item()),
        "ssim": float(ssim(pred, gt).item()),
    }
    for net in ("vgg", "alex"):
        out[f"lpips_{net}"] = float(
            lpips_value(pred, gt, net, lpips_models.get(net)).item())
    return out

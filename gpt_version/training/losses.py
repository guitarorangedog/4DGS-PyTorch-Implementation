"""Image reconstruction losses for static 3D-GS training [INFRA].

Implements the 3D-GS photometric loss (Commit 8 scope: static only, no
TV/temporal terms — those arrive with 4D regularization in Commit 15):

- ``L1 = mean(|pred - gt|)`` (official ``loss_utils.l1_loss``).
- ``SSIM`` with the official 11x11 Gaussian window (sigma 1.5),
  ``C1 = 0.01^2``, ``C2 = 0.03^2`` (official ``loss_utils.ssim``).
- Combined: ``L = (1 - λ) * L1 + λ * (1 - SSIM)``.

Formulas match ``utils/loss_utils.py`` of 3D-GS / 4DGaussians exactly.
Documented deviation: official ``create_window`` wraps the window in the
deprecated ``torch.autograd.Variable`` and moves it with
``window.cuda(img.get_device())``; here the window is a plain tensor placed
via ``window.to(img.device).type_as(img)`` — same values, device-agnostic
so CPU tests run. No ``lpips`` dependency in this module.
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["l1_loss", "ssim", "reconstruction_loss", "reconstruction_loss_4dgs"]


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over all pixels and channels."""
    return torch.abs(pred - gt).mean()


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([
        math.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
        for x in range(window_size)
    ])
    return (gauss / gauss.sum()).unsqueeze(1)


def _ssim_window(window_size: int, channel: int, dtype, device) -> torch.Tensor:
    window_1d = _gaussian_window(window_size, 1.5)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous().to(
        device=device, dtype=dtype
    )


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity, mean over pixels/channels (official formula).

    Args:
        img1: ``[B, C, H, W]`` rendered images in ``[0, 1]``.
        img2: ``[B, C, H, W]`` ground truth, same shape.
        window_size: Gaussian window size (official default 11).

    Returns:
        Scalar SSIM in ``(-1, 1]`` (1.0 = identical).
    """
    assert img1.shape == img2.shape, f"{tuple(img1.shape)} vs {tuple(img2.shape)}"
    channel = img1.size(-3)
    window = _ssim_window(window_size, channel, img1.dtype, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def reconstruction_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float = 0.0) -> torch.Tensor:
    """Combined static loss: ``(1 - λ) * L1 + λ * (1 - SSIM)``.

    The canonical 3D-GS weighting (3D-GS used ``λ = 0.2``). With the official
    4DGaussians default ``λ = 0`` this reduces to pure ``L1``.

    Note: official 4DGaussians ``train.py`` instead computes
    ``L1 + λ * (1 - SSIM)`` (full L1 kept). The two coincide at ``λ = 0``
    (the default) and differ only when ``λ > 0``; we use the canonical
    3D-GS form per the Commit 8 spec.
    """
    assert 0.0 <= lambda_dssim <= 1.0
    l1 = l1_loss(pred, gt)
    if lambda_dssim == 0.0:
        return l1
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim(pred, gt))


def reconstruction_loss_4dgs(pred: torch.Tensor, gt: torch.Tensor,
                             lambda_dssim: float = 0.0) -> torch.Tensor:
    """Official 4DGaussians reconstruction loss: ``L1 + λ * (1 - SSIM)``.

    Verbatim ``train.py`` semantics (NOT the canonical ``(1-λ)`` form above):
    the full L1 term is always kept and the DSSIM term is added on top.
    Official default ``λ = 0`` reduces both forms to pure ``L1``. The Commit
    16 fine stage must use THIS form.
    """
    assert 0.0 <= lambda_dssim <= 1.0
    l1 = l1_loss(pred, gt)
    if lambda_dssim == 0.0:
        return l1
    return l1 + lambda_dssim * (1.0 - ssim(pred, gt))

"""Boundary-aware and Focal-Tversky loss functions for NSD and IoU optimization."""

import torch
import torch.nn.functional as F


def _morphological_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract soft morphological contour band via dilation minus erosion."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(dilated - eroded, 0.0, 1.0)


def compute_boundary_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute soft Dice + BCE loss focused on the 3-pixel contour band."""
    p = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    pred_bound = _morphological_boundary(p, kernel_size=3)
    gt_bound = _morphological_boundary(target, kernel_size=3)

    b = p.shape[0]
    pb, gb = pred_bound.view(b, -1), gt_bound.view(b, -1)
    inter = torch.sum(pb * gb, dim=1)
    card = torch.sum(pb + gb, dim=1)
    bound_dice = torch.mean(1.0 - (2.0 * inter + smooth) / (card + smooth))

    contour_band = _morphological_boundary(target, kernel_size=5)
    bce_map = F.binary_cross_entropy(p, target, reduction="none")
    band_bce = torch.sum(bce_map * contour_band) / (torch.sum(contour_band) + 1e-6)
    return 0.5 * bound_dice + 0.5 * band_bce


def compute_focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.4,
    beta: float = 0.6,
    gamma: float = 1.33,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute Focal-Tversky loss to improve IoU and small-nodule boundary recall."""
    b = pred.shape[0]
    p, t = pred.view(b, -1), target.view(b, -1)
    tp = torch.sum(p * t, dim=1)
    fp = torch.sum(p * (1.0 - t), dim=1)
    fn = torch.sum((1.0 - p) * t, dim=1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return torch.mean((1.0 - tversky) ** gamma)

"""Distance-weighted boundary, Soft-IoU (Jaccard), and Focal-Tversky loss functions."""

import torch
import torch.nn.functional as F


def _morphological_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Extract soft morphological contour band via dilation minus erosion."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return torch.clamp(dilated - eroded, 0.0, 1.0)


def compute_boundary_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute multi-shell distance-decayed boundary BCE + soft contour Dice loss."""
    p = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
    pred_bound = _morphological_boundary(p, kernel_size=3)
    gt_bound = _morphological_boundary(target, kernel_size=3)

    b = p.shape[0]
    pb, gb = pred_bound.view(b, -1), gt_bound.view(b, -1)
    bound_dice = torch.mean(1.0 - (2.0 * torch.sum(pb * gb, dim=1) + smooth) / (torch.sum(pb + gb, dim=1) + smooth))

    shell_1 = gt_bound
    shell_2 = _morphological_boundary(target, kernel_size=5)
    shell_3 = _morphological_boundary(target, kernel_size=7)
    dist_weight = 1.0 + 1.2 * shell_1 + 0.6 * shell_2 + 0.3 * shell_3
    bce_map = F.binary_cross_entropy(p, target, reduction="none")
    weighted_bce = torch.mean(bce_map * dist_weight)
    return 0.5 * bound_dice + 0.5 * weighted_bce


def compute_soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute direct differentiable Jaccard (Soft-IoU) loss per slice."""
    b = pred.shape[0]
    p, t = pred.view(b, -1), target.view(b, -1)
    inter = torch.sum(p * t, dim=1)
    union = torch.sum(p + t - p * t, dim=1)
    return torch.mean(1.0 - (inter + smooth) / (union + smooth))


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

"""Loss functions for MUS-Net multi-window guidance with Dice support."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mus_net.boundary_loss import compute_boundary_loss, compute_focal_tversky_loss


class MultiWindowSegmentationLoss(nn.Module):
    """Calculates composite BCE + Dice loss (Eq. 24) to resolve class imbalance."""

    def __init__(
        self,
        alpha_1: float = 1.0,
        alpha_2: float = 0.5,
        use_dice_loss: bool = True,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        use_posterior_kl: bool = False,
        kl_weight: float = 0.001,
        boundary_weight: float = 0.0,
        tversky_weight: float = 0.0,
        use_uncertainty_aux: bool = False,
    ) -> None:
        """Initialize loss weights and flags."""
        super().__init__()
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.use_dice_loss, self.dice_weight, self.bce_weight = use_dice_loss, dice_weight, bce_weight
        self.use_posterior_kl, self.kl_weight = use_posterior_kl, kl_weight
        self.boundary_weight, self.tversky_weight, self.use_uncertainty_aux = (
            boundary_weight, tversky_weight, use_uncertainty_aux
        )

    @staticmethod
    def _compute_dice_loss(
        pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None, smooth: float = 1.0
    ) -> torch.Tensor:
        """Calculate soft Dice loss per slice across spatial dimensions (H, W)."""
        b = pred.shape[0]
        w = weight.view(b, -1) if weight is not None else 1.0
        p_f, t_f = pred.view(b, -1) * w, target.view(b, -1) * w
        dice = (2.0 * torch.sum(p_f * t_f, dim=1) + smooth) / (torch.sum(p_f + t_f, dim=1) + smooth)
        return torch.mean(1.0 - dice)

    def _compute_term_loss(
        self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Combine BCE with soft Dice loss to tackle 98% background imbalance."""
        clamped_pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        if weight is not None:
            bce = (F.binary_cross_entropy(clamped_pred, target, reduction="none") * weight).mean()
        else:
            bce = F.binary_cross_entropy(clamped_pred, target)
        if self.use_dice_loss:
            return self.bce_weight * bce + self.dice_weight * self._compute_dice_loss(clamped_pred, target, weight)
        return bce

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite objective function across all active branches."""
        final_pred = model_outputs["final_prediction"]
        loss_ffm = self._compute_term_loss(final_pred, ground_truth_mask)
        if self.boundary_weight > 0.0:
            loss_ffm = loss_ffm + self.boundary_weight * compute_boundary_loss(final_pred, ground_truth_mask)
        if self.tversky_weight > 0.0:
            loss_ffm = loss_ffm + self.tversky_weight * compute_focal_tversky_loss(final_pred, ground_truth_mask)
        branch_losses_sum = torch.tensor(0.0, device=ground_truth_mask.device)
        loss_details = {"loss_final": float(loss_ffm.item())}
        weights_map = model_outputs.get("window_weights") or {}

        active_count = 0
        for branch_name, mask_pred in model_outputs.get("branch_masks", {}).items():
            if mask_pred is not None:
                pw = (0.25 + 0.75 * weights_map["medias"].clamp(0.0, 1.0)).detach() if (
                    self.use_uncertainty_aux and branch_name == "medias" and "medias" in weights_map
                ) else None
                term = self._compute_term_loss(mask_pred, ground_truth_mask, pw)
                branch_losses_sum = branch_losses_sum + term
                loss_details[f"loss_{branch_name}"] = float(term.item())
                active_count += 1

        total_loss = self.alpha_1 * loss_ffm
        if active_count > 0:
            total_loss = total_loss + self.alpha_2 * branch_losses_sum

        if self.use_posterior_kl and "latent_parameters" in model_outputs:
            kl_sum = torch.tensor(0.0, device=ground_truth_mask.device)
            for _, (mu, log_var) in model_outputs["latent_parameters"].items():
                kl = -0.5 * torch.mean(torch.sum(1.0 + log_var - mu**2 - torch.exp(log_var), dim=1))
                kl_sum = kl_sum + kl
            total_loss = total_loss + self.kl_weight * kl_sum
            loss_details["loss_kl"] = float(kl_sum.item())

        loss_details["total_loss"] = float(total_loss.item())
        return total_loss, loss_details

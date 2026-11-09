"""Loss functions for MUS-Net multi-window guidance with Dice support."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> None:
        """Initialize loss weights and flags."""
        super().__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.use_posterior_kl = use_posterior_kl
        self.kl_weight = kl_weight

    @staticmethod
    def _compute_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """Calculate soft Dice loss preventing collapse to all-background predictions."""
        intersection = torch.sum(pred * target)
        cardinality = torch.sum(pred + target)
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        return 1.0 - dice

    def _compute_term_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combine BCE with soft Dice loss to tackle 98% background imbalance."""
        bce = F.binary_cross_entropy(pred, target)
        if self.use_dice_loss:
            dice = self._compute_dice_loss(pred, target)
            return self.bce_weight * bce + self.dice_weight * dice
        return bce

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite objective function across all active branches."""
        loss_ffm = self._compute_term_loss(model_outputs["final_prediction"], ground_truth_mask)
        branch_losses_sum = torch.tensor(0.0, device=ground_truth_mask.device)
        loss_details = {"loss_final": float(loss_ffm.item())}

        active_count = 0
        for branch_name, mask_pred in model_outputs.get("branch_masks", {}).items():
            if mask_pred is not None:
                term = self._compute_term_loss(mask_pred, ground_truth_mask)
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

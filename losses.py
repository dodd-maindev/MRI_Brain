"""Loss functions for MUS-Net multi-window guidance."""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiWindowSegmentationLoss(nn.Module):
    """Calculates composite BCE loss (Eq. 24) with optional CVAE KL divergence."""

    def __init__(
        self,
        alpha_1: float = 1.0,
        alpha_2: float = 0.5,
        use_posterior_kl: bool = False,
        kl_weight: float = 0.001,
    ) -> None:
        """Initialize loss weights and flags."""
        super().__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.use_posterior_kl = use_posterior_kl
        self.kl_weight = kl_weight

    @staticmethod
    def _compute_kl_divergence(mean: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence between predicted latent Gaussian and standard Normal."""
        return -0.5 * torch.mean(
            torch.sum(1.0 + log_variance - mean**2 - torch.exp(log_variance), dim=1)
        )

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        ground_truth_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite objective function."""
        final_prediction = model_outputs["final_prediction"]
        loss_ffm = F.binary_cross_entropy(final_prediction, ground_truth_mask)

        branch_losses_sum = torch.tensor(0.0, device=ground_truth_mask.device)
        loss_details = {"loss_final": float(loss_ffm.item())}

        branch_masks = model_outputs.get("branch_masks", {})
        active_branch_count = 0
        for branch_name, mask_pred in branch_masks.items():
            if mask_pred is not None:
                bce = F.binary_cross_entropy(mask_pred, ground_truth_mask)
                branch_losses_sum = branch_losses_sum + bce
                loss_details[f"loss_{branch_name}"] = float(bce.item())
                active_branch_count += 1

        total_loss = self.alpha_1 * loss_ffm
        if active_branch_count > 0:
            total_loss = total_loss + self.alpha_2 * branch_losses_sum

        if self.use_posterior_kl and "latent_parameters" in model_outputs:
            kl_sum = torch.tensor(0.0, device=ground_truth_mask.device)
            for _, (mu, log_var) in model_outputs["latent_parameters"].items():
                kl_sum = kl_sum + self._compute_kl_divergence(mu, log_var)
            total_loss = total_loss + self.kl_weight * kl_sum
            loss_details["loss_kl"] = float(kl_sum.item())

        loss_details["total_loss"] = float(total_loss.item())
        return total_loss, loss_details


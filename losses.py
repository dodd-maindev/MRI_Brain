"""Loss functions for MUS-Net, AMU-Net, and Evidential EDL-MUS-Net."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mus_net.boundary_loss import compute_boundary_loss, compute_focal_tversky_loss, compute_soft_iou_loss
from mus_net.evidential_loss import compute_evidential_beta_loss


class MultiWindowSegmentationLoss(nn.Module):
    """Calculates BCE/Evidential-Beta + Dice + Soft-IoU + Distance-Boundary loss."""

    def __init__(
        self, alpha_1: float = 1.0, alpha_2: float = 0.5, use_dice_loss: bool = True,
        dice_weight: float = 1.0, bce_weight: float = 1.0, use_posterior_kl: bool = False,
        kl_weight: float = 0.001, boundary_weight: float = 0.0, tversky_weight: float = 0.0,
        use_uncertainty_aux: bool = False, iou_weight: float = 0.0,
        use_evidential: bool = False, edl_kl_weight: float = 0.05,
    ) -> None:
        """Initialize loss weights and evidential flags."""
        super().__init__()
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.use_dice_loss, self.dice_weight, self.bce_weight = use_dice_loss, dice_weight, bce_weight
        self.use_posterior_kl, self.kl_weight = use_posterior_kl, kl_weight
        self.boundary_weight, self.tversky_weight = boundary_weight, tversky_weight
        self.use_uncertainty_aux, self.iou_weight = use_uncertainty_aux, iou_weight
        self.use_evidential, self.edl_kl_weight = use_evidential, edl_kl_weight

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
        """Combine BCE with soft Dice loss to tackle background imbalance."""
        clamped_pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        bce = (
            (F.binary_cross_entropy(clamped_pred, target, reduction="none") * weight).mean()
            if weight is not None else F.binary_cross_entropy(clamped_pred, target)
        )
        if self.use_dice_loss:
            return self.bce_weight * bce + self.dice_weight * self._compute_dice_loss(clamped_pred, target, weight)
        return bce

    def forward(
        self, model_outputs: Dict[str, Any], ground_truth_mask: torch.Tensor, annealing_factor: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite objective across final head and active window branches."""
        final_pred = model_outputs["final_prediction"]
        if self.use_evidential and model_outputs.get("final_evidence") is not None:
            a_f, b_f = model_outputs["final_evidence"]
            loss_ffm = compute_evidential_beta_loss(a_f, b_f, ground_truth_mask, self.edl_kl_weight, annealing_factor)
            if self.use_dice_loss:
                loss_ffm = loss_ffm + self.dice_weight * self._compute_dice_loss(final_pred, ground_truth_mask)
        else:
            loss_ffm = self._compute_term_loss(final_pred, ground_truth_mask)

        if self.boundary_weight > 0.0:
            loss_ffm = loss_ffm + self.boundary_weight * compute_boundary_loss(final_pred, ground_truth_mask)
        if self.tversky_weight > 0.0:
            loss_ffm = loss_ffm + self.tversky_weight * compute_focal_tversky_loss(final_pred, ground_truth_mask)
        if self.iou_weight > 0.0:
            loss_ffm = loss_ffm + self.iou_weight * compute_soft_iou_loss(final_pred, ground_truth_mask)

        branch_sum = torch.tensor(0.0, device=ground_truth_mask.device)
        loss_details, active_count = {"loss_final": float(loss_ffm.item())}, 0
        if self.use_evidential and model_outputs.get("branch_evidence"):
            for name, (a_b, b_b) in model_outputs["branch_evidence"].items():
                term = compute_evidential_beta_loss(a_b, b_b, ground_truth_mask, self.edl_kl_weight, annealing_factor)
                branch_sum, active_count = branch_sum + term, active_count + 1
                loss_details[f"loss_{name}"] = float(term.item())
        else:
            w_map = model_outputs.get("window_weights") or {}
            for name, mask_pred in model_outputs.get("branch_masks", {}).items():
                if mask_pred is not None:
                    pw = (0.25 + 0.75 * w_map["medias"].clamp(0.0, 1.0)).detach() if (
                        self.use_uncertainty_aux and name == "medias" and "medias" in w_map
                    ) else None
                    term = self._compute_term_loss(mask_pred, ground_truth_mask, pw)
                    branch_sum, active_count = branch_sum + term, active_count + 1
                    loss_details[f"loss_{name}"] = float(term.item())

        total_loss = self.alpha_1 * loss_ffm + (self.alpha_2 * branch_sum if active_count > 0 else 0.0)
        if self.use_posterior_kl and "latent_parameters" in model_outputs:
            kl_sum = sum(
                -0.5 * torch.mean(torch.sum(1.0 + lv - mu**2 - torch.exp(lv), dim=1))
                for _, (mu, lv) in model_outputs["latent_parameters"].items()
            )
            total_loss = total_loss + self.kl_weight * kl_sum
        loss_details["total_loss"] = float(total_loss.item())
        return total_loss, loss_details

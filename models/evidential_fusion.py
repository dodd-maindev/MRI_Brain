"""Evidential Subjective Logic & Dempster-Shafer Multi-Window Fusion (EDL-MWF)."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

Opinion = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class EvidentialProjectionHead(nn.Module):
    """Maps feature maps to non-negative Beta evidence (e+, e-) and Subjective Opinion."""

    def __init__(self, in_channels: int) -> None:
        """Initialize 1x1 evidence projection layer."""
        super().__init__()
        self.evidence_conv = nn.Conv2d(in_channels, 2, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Opinion]:
        """Compute Beta parameters (alpha, beta) and Opinion (b+, b-, u)."""
        evidence = torch.clamp(F.softplus(self.evidence_conv(features)), max=50.0)
        e_pos, e_neg = evidence[:, 0:1], evidence[:, 1:2]
        alpha, beta = e_pos + 1.0, e_neg + 1.0
        strength = alpha + beta
        opinion = (e_pos / strength, e_neg / strength, 2.0 / strength)
        return alpha, beta, opinion


def combine_two_opinions(op1: Opinion, op2: Opinion) -> Tuple[Opinion, torch.Tensor]:
    """Combine two binary Subjective Logic opinions via closed-form Dempster-Shafer rule."""
    b1_p, b1_n, u1 = op1
    b2_p, b2_n, u2 = op2
    conflict = b1_p * b2_n + b1_n * b2_p
    norm = torch.clamp(1.0 - conflict, min=1e-6)
    b_p = (b1_p * b2_p + b1_p * u2 + b2_p * u1) / norm
    b_n = (b1_n * b2_n + b1_n * u2 + b2_n * u1) / norm
    u = torch.clamp((u1 * u2) / norm, min=1e-5, max=1.0)
    return (b_p, b_n, u), conflict


def opinion_to_beta_and_prob(opinion: Opinion) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover Beta parameters (alpha, beta) and expected probability p = b+ + 0.5*u."""
    b_p, b_n, u = opinion
    strength = 2.0 / torch.clamp(u, min=1e-5)
    alpha = b_p * strength + 1.0
    beta = b_n * strength + 1.0
    prob = torch.clamp(b_p + 0.5 * u, 1e-6, 1.0 - 1e-6)
    return alpha, beta, prob


class DempsterShaferWindowFusion(nn.Module):
    """Parameter-free Dempster-Shafer multi-window fusion and epistemic feature modulation."""

    def __init__(self, active_branches: List[str]) -> None:
        """Store active branch ordering."""
        super().__init__()
        self.active_branches = active_branches

    def forward(
        self,
        y_features: List[torch.Tensor],
        f_features: List[torch.Tensor],
        opinions: Dict[str, Opinion],
    ) -> Tuple[torch.Tensor, Opinion, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Fuse branch opinions via Dempster-Shafer and modulate features by (1 - u_i)."""
        fused_op = opinions[self.active_branches[0]]
        for b in self.active_branches[1:]:
            fused_op, _ = combine_two_opinions(fused_op, opinions[b])

        conflict_lm = None
        if "lung" in opinions and "medias" in opinions:
            _, conflict_lm = combine_two_opinions(opinions["lung"], opinions["medias"])

        confidences = [torch.clamp(1.0 - opinions[b][2], min=0.0) + 1e-4 for b in self.active_branches]
        conf_sum = sum(confidences)
        weights = [c / conf_sum for c in confidences]

        gated_y = [y * (1.0 + w) for y, w in zip(y_features, weights)]
        gated_f = [f * (1.0 + w) for f, w in zip(f_features, weights)]
        weight_dict = {b: w for b, w in zip(self.active_branches, weights)}
        return torch.cat(gated_y + gated_f, dim=1), fused_op, weight_dict, conflict_lm

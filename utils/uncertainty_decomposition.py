"""Information-theoretic uncertainty decomposition for Beta evidential predictions."""

from typing import Dict
import torch


def decompose_beta_uncertainty(alpha: torch.Tensor, beta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Decompose Beta(alpha, beta) predictive entropy into aleatoric and epistemic (MI)."""
    eps = 1e-7
    strength = torch.clamp(alpha + beta, min=2.0)
    prob = torch.clamp(alpha / strength, min=eps, max=1.0 - eps)

    total = -(prob * torch.log(prob) + (1.0 - prob) * torch.log(1.0 - prob))
    aleatoric = (
        torch.digamma(strength + 1.0)
        - prob * torch.digamma(alpha + 1.0)
        - (1.0 - prob) * torch.digamma(beta + 1.0)
    )
    aleatoric = torch.clamp(aleatoric, min=0.0)
    epistemic = torch.clamp(total - aleatoric, min=0.0)
    vacuity = 2.0 / strength
    return {
        "total": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "vacuity": vacuity,
    }

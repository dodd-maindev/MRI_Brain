"""Exact Evidential Bayes Risk (digamma) and Beta KL divergence loss."""

import torch


def compute_evidential_beta_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    target: torch.Tensor,
    kl_weight: float = 0.05,
    annealing_factor: float = 1.0,
) -> torch.Tensor:
    """Compute Type-II ML expected cross-entropy + closed-form Beta(a~, b~) || Beta(1, 1) KL."""
    y = target.float()
    strength = alpha + beta
    # Expected negative log-likelihood under p ~ Beta(alpha, beta)
    digamma_s = torch.digamma(strength)
    expected_ce = y * (digamma_s - torch.digamma(alpha)) + (1.0 - y) * (digamma_s - torch.digamma(beta))

    # Remove non-misleading evidence so KL only penalizes wrong-class evidence toward Beta(1, 1)
    a_tilde = y + (1.0 - y) * alpha
    b_tilde = (1.0 - y) + y * beta
    s_tilde = a_tilde + b_tilde
    kl_beta = (
        torch.lgamma(s_tilde) - torch.lgamma(a_tilde) - torch.lgamma(b_tilde)
        + (a_tilde - 1.0) * (torch.digamma(a_tilde) - torch.digamma(s_tilde))
        + (b_tilde - 1.0) * (torch.digamma(b_tilde) - torch.digamma(s_tilde))
    )
    return torch.mean(expected_ce + (kl_weight * annealing_factor) * kl_beta)

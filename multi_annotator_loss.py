"""Multi-annotator Beta-Binomial, Soft-Label evidential losses, and supervision strategy."""

from typing import Any, Dict, Optional, Tuple, Union
import torch


def _misleading_beta_kl(alpha: torch.Tensor, beta: torch.Tensor, y_hard: torch.Tensor) -> torch.Tensor:
    """Compute closed-form Beta(a_tilde, b_tilde) || Beta(1, 1) KL on hard labels."""
    a_t = y_hard + (1.0 - y_hard) * alpha
    b_t = (1.0 - y_hard) + y_hard * beta
    s_t = a_t + b_t
    return (
        torch.lgamma(s_t) - torch.lgamma(a_t) - torch.lgamma(b_t)
        + (a_t - 1.0) * (torch.digamma(a_t) - torch.digamma(s_t))
        + (b_t - 1.0) * (torch.digamma(b_t) - torch.digamma(s_t))
    )


def compute_beta_binomial_loss(
    alpha: torch.Tensor, beta: torch.Tensor, vote_pos: torch.Tensor, vote_total: torch.Tensor,
    kl_weight: float = 0.05, annealing_factor: float = 1.0,
) -> torch.Tensor:
    """Compute normalized Beta-Binomial NLL + unanimous-pixel misleading-evidence KL."""
    y, n = vote_pos.float(), vote_total.float().clamp(min=1.0)
    log_comb = torch.lgamma(n + 1.0) - torch.lgamma(y + 1.0) - torch.lgamma(n - y + 1.0)
    log_b_num = torch.lgamma(y + alpha) + torch.lgamma(n - y + beta) - torch.lgamma(n + alpha + beta)
    log_b_den = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    nll = -(log_comb + log_b_num - log_b_den) / n

    unanimous = ((y <= 0.0) | (y >= n)).float()
    kl = unanimous * _misleading_beta_kl(alpha, beta, (y >= n).float())
    return torch.mean(nll + (kl_weight * annealing_factor) * kl)


def compute_soft_evidential_loss(
    alpha: torch.Tensor, beta: torch.Tensor, vote_pos: torch.Tensor, vote_total: torch.Tensor,
    kl_weight: float = 0.05, annealing_factor: float = 1.0,
) -> torch.Tensor:
    """Compute soft-label Expected-CE + unanimous-pixel misleading-evidence KL."""
    y, n = vote_pos.float(), vote_total.float().clamp(min=1.0)
    y_bar = torch.clamp(y / n, 0.0, 1.0)
    dig_s = torch.digamma(alpha + beta)
    ece = y_bar * (dig_s - torch.digamma(alpha)) + (1.0 - y_bar) * (dig_s - torch.digamma(beta))

    unanimous = ((y <= 0.0) | (y >= n)).float()
    kl = unanimous * _misleading_beta_kl(alpha, beta, (y >= n).float())
    return torch.mean(ece + (kl_weight * annealing_factor) * kl)


class SupervisionTargetResolver:
    """Resolves segmentation targets and evidential loss terms by supervision mode."""

    def __init__(self, supervision: str = "ann0", edl_kl_weight: float = 0.05) -> None:
        """Initialize supervision mode ('ann0', 'random_annotator', 'soft_label', 'beta_binomial')."""
        self.supervision = supervision
        self.edl_kl_weight = edl_kl_weight

    def resolve(
        self, target_or_batch: Union[torch.Tensor, Dict[str, Any]], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (soft_target, boundary_target, vote_positive, vote_total) on device."""
        if isinstance(target_or_batch, torch.Tensor):
            t = target_or_batch.to(device)
            return t, t, None, None
        if self.supervision == "random_annotator":
            t = target_or_batch.get("mask_random", target_or_batch["mask"]).to(device)
            return t, t, None, None
        if self.supervision in ("soft_label", "beta_binomial") and "vote_positive" in target_or_batch:
            vp = target_or_batch["vote_positive"].to(device)
            vt = target_or_batch["vote_total"].to(device).clamp(min=1.0)
            y_bar = torch.clamp(vp / vt, 0.0, 1.0)
            return y_bar, (y_bar >= 0.5).float(), vp, vt
        t = target_or_batch["mask"].to(device)
        return t, t, None, None

    def compute_evidential_term(
        self, alpha: torch.Tensor, beta: torch.Tensor, soft_target: torch.Tensor,
        vote_pos: Optional[torch.Tensor], vote_total: Optional[torch.Tensor],
        annealing_factor: float, fallback_fn: Any,
    ) -> torch.Tensor:
        """Dispatch to Beta-Binomial, Soft-Label, or standard Bayes-risk evidential loss."""
        if self.supervision == "beta_binomial" and vote_pos is not None and vote_total is not None:
            return compute_beta_binomial_loss(alpha, beta, vote_pos, vote_total, self.edl_kl_weight, annealing_factor)
        if self.supervision == "soft_label" and vote_pos is not None and vote_total is not None:
            return compute_soft_evidential_loss(alpha, beta, vote_pos, vote_total, self.edl_kl_weight, annealing_factor)
        return fallback_fn(alpha, beta, soft_target, self.edl_kl_weight, annealing_factor)

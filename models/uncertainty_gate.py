"""MWM-Guided Cross-Window Dynamic Uncertainty Gate for multi-window fusion."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class CrossWindowUncertaintyGate(nn.Module):
    """Computes pixel-wise branch entropy and Soft-MWM to dynamically weight branches."""

    def __init__(self, feature_channels: int, active_branches: List[str]) -> None:
        """Initialize lightweight spatial gating network over active branches."""
        super().__init__()
        self.active_branches = active_branches
        num_branches = len(active_branches)
        # Input: filtered features y_i + branch entropies U_i + 3-channel Soft-MWM
        gate_in_channels = num_branches * feature_channels + num_branches + 3
        hidden_channels = max(16, feature_channels)
        self.gate_net = nn.Sequential(
            nn.Conv2d(gate_in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_branches, kernel_size=1),
        )

    @staticmethod
    def compute_entropy(prob_map: torch.Tensor) -> torch.Tensor:
        """Compute normalized binary predictive entropy in [0, 1] from mask S_i."""
        p = torch.clamp(prob_map, 1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return entropy / 0.6931471805599453

    @staticmethod
    def compute_soft_mwm(masks: Dict[str, Optional[torch.Tensor]], ref: torch.Tensor) -> torch.Tensor:
        """Compute continuous 3-channel Soft-MWM [medias-only, lung-only, common]."""
        s_lung = masks.get("lung")
        s_medias = masks.get("medias")
        if s_lung is None or s_medias is None:
            b, _, h, w = ref.shape
            return torch.zeros((b, 3, h, w), device=ref.device, dtype=ref.dtype)
        common = s_lung * s_medias
        lung_only = s_lung * (1.0 - s_medias)
        medias_only = s_medias * (1.0 - s_lung)
        return torch.cat([medias_only, lung_only, common], dim=1)

    def forward(
        self,
        y_features: List[torch.Tensor],
        f_features: List[torch.Tensor],
        masks: Dict[str, Optional[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Weight each branch's y_i and f_i features using uncertainty-aware spatial gates."""
        entropies = [
            self.compute_entropy(masks[b]) if masks.get(b) is not None
            else torch.zeros_like(y_features[0][:, :1])
            for b in self.active_branches
        ]
        soft_mwm = self.compute_soft_mwm(masks, y_features[0])
        gate_input = torch.cat(y_features + entropies + [soft_mwm], dim=1)
        weights = torch.softmax(self.gate_net(gate_input), dim=1)

        gated_y = [y * (1.0 + weights[:, i : i + 1]) for i, y in enumerate(y_features)]
        gated_f = [f * (1.0 + weights[:, i : i + 1]) for i, f in enumerate(f_features)]
        weight_dict = {b: weights[:, i : i + 1] for i, b in enumerate(self.active_branches)}
        return torch.cat(gated_y + gated_f, dim=1), weight_dict, soft_mwm

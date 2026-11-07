"""Probabilistic Attention U-Net combining Attention U-Net and CVAE PriorNet."""

from typing import Tuple
import torch
import torch.nn as nn
from mus_net.models.attention_unet import AttentionUNet
from mus_net.models.prior_net import PriorNet


class ProbabilisticAttentionUNet(nn.Module):
    """FEM branch extracting deterministic features r_i and latent variants z_i."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dimension: int = 6,
    ) -> None:
        """Initialize Attention U-Net and PriorNet."""
        super().__init__()
        self.attention_unet = AttentionUNet(
            in_channels=in_channels, base_channels=base_channels
        )
        self.prior_net = PriorNet(
            in_channels=in_channels, latent_dimension=latent_dimension
        )
        self.out_channels = base_channels + latent_dimension

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract r_i, sample z_i, broadcast, and concatenate to form f_i."""
        feature_maps = self.attention_unet(input_tensor)
        latent_vector, mean, log_variance = self.prior_net.sample(input_tensor)

        height, width = feature_maps.shape[2], feature_maps.shape[3]
        broadcast_latent = latent_vector.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, height, width
        )
        fused_features = torch.cat([feature_maps, broadcast_latent], dim=1)
        return fused_features, mean, log_variance


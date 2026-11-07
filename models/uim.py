"""Uncertainty Indication Module (UIM) and Multi-Window Mask (MWM) generator."""

from typing import Tuple
import torch
import torch.nn as nn
from mus_net.models.unet_blocks import ConvolutionalBlock


class UncertaintyIndicationModule(nn.Module):
    """Generates uncertainty-aware features F_i and preliminary mask S_i."""

    def __init__(self, in_channels: int, intermediate_channels: int) -> None:
        """Initialize intermediate feature extractor and mask projection heads."""
        super().__init__()
        self.feature_conv = ConvolutionalBlock(in_channels, intermediate_channels)
        self.mask_conv = ConvolutionalBlock(intermediate_channels, intermediate_channels)
        self.final_head = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intermediate feature map F_i and preliminary mask S_i."""
        intermediate_features = self.feature_conv(input_features)
        deep_features = self.mask_conv(intermediate_features)
        mask_prediction = self.final_head(deep_features)
        return intermediate_features, mask_prediction


def compute_multi_window_mask(
    lung_mask: torch.Tensor,
    mediastinal_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Compute 3-channel MWM: common region, lung-only (green), medias-only (red)."""
    binary_lung = (lung_mask >= threshold).float()
    binary_medias = (mediastinal_mask >= threshold).float()

    common_region = binary_lung * binary_medias
    lung_only_region = binary_lung * (1.0 - binary_medias)
    medias_only_region = binary_medias * (1.0 - binary_lung)

    # Channel order: Red = Medias-only, Green = Lung-only, Blue = Common overlap
    return torch.cat(
        [medias_only_region, lung_only_region, common_region], dim=1
    )


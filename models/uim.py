"""Uncertainty Indication Module (UIM) and Multi-Window Mask (MWM) generator."""

from typing import Tuple
import torch
import torch.nn as nn
from mus_net.models.unet_blocks import ConvolutionalBlock


class UncertaintyIndicationModule(nn.Module):
    """Generates uncertainty-aware features F_i and preliminary mask S_i (Sec 3.3)."""

    def __init__(self, in_channels: int, intermediate_channels: int) -> None:
        """Initialize single-layer intermediate feature conv and mask prediction conv."""
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, input_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute intermediate feature map F_i and preliminary mask S_i."""
        intermediate_features = self.feature_conv(input_features)
        mask_prediction = self.mask_conv(intermediate_features)
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


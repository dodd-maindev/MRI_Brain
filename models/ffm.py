"""Feature Filter Module (FFM) with spatial self-attention and domain filters."""

import math
from typing import Optional
import torch
import torch.nn as nn
from mus_net.models.filters import (
    SobelFilter,
    OtsuThresholdFilter,
    GaborFilterBankModule,
    LaplacianFilter,
)


class SpatialSelfAttention(nn.Module):
    """Spatial self-attention over H*W tokens with optional key-value pooling."""

    def __init__(self, in_channels: int, downsample_ratio: int = 4) -> None:
        """Initialize 1x1 projection convolutions and scaling factor."""
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = 1.0 / math.sqrt(in_channels)
        self.downsample = (
            nn.AvgPool2d(downsample_ratio, downsample_ratio)
            if downsample_ratio > 1
            else nn.Identity()
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Compute spatial self-attention map and aggregate value representations."""
        b, c, h, w = input_features.shape
        q = self.query_conv(input_features).view(b, c, h * w).permute(0, 2, 1)
        kv_down = self.downsample(input_features)
        k = self.key_conv(kv_down).view(b, c, -1)
        v = self.value_conv(kv_down).view(b, c, -1).permute(0, 2, 1)

        energy = torch.bmm(q, k) * self.scale
        attention_map = torch.softmax(energy, dim=-1)
        out = torch.bmm(attention_map, v)
        return out.permute(0, 2, 1).contiguous().view(b, c, h, w)


class FeatureFilterBranch(nn.Module):
    """Executes Self-Attention, applies specified filter, and blends: y_i = beta * filter + x_i."""

    def __init__(
        self,
        channels: int,
        filter_type: str = "sobel",
        beta: float = 1.0,
        downsample_ratio: int = 4,
    ) -> None:
        """Initialize attention and domain-specific filter."""
        super().__init__()
        self.beta = beta
        self.attention = SpatialSelfAttention(channels, downsample_ratio)
        filter_map = {
            "sobel": SobelFilter(),
            "otsu": OtsuThresholdFilter(),
            "gabor": GaborFilterBankModule(channels),
            "laplacian": LaplacianFilter(),
            "none": nn.Identity(),
        }
        self.filter = filter_map.get(filter_type.lower(), SobelFilter())

    def forward(self, intermediate_features: torch.Tensor) -> torch.Tensor:
        """Transform features: d_i = Attention(F_i), then y_i = beta * filter(d_i) + F_i."""
        attended = self.attention(intermediate_features)
        filtered = self.filter(attended)
        # ASSUMPTION: Eq. 21 combines weighted filtered features with input features
        return self.beta * filtered + intermediate_features


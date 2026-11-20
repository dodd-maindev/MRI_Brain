"""Multi-Scale Atrous Spatial Pyramid Pooling (ASPP) bottleneck for 64x64 patches."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AtrousConvBnRelu(nn.Module):
    """Single dilated convolution block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        """Initialize dilated conv-bn-relu."""
        super().__init__()
        padding = dilation if kernel_size > 1 else 0
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated convolution block."""
        return self.block(x)


class MultiScaleASPPBottleneck(nn.Module):
    """4-branch ASPP bottleneck with Spatial Dropout to preserve 4x4 geometry."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.15) -> None:
        """Initialize 1x1, dilated 3x3 (rates 2, 4), and image-pooling branches."""
        super().__init__()
        branch_ch = max(32, out_channels // 4)
        self.b0 = _AtrousConvBnRelu(in_channels, branch_ch, kernel_size=1, dilation=1)
        self.b1 = _AtrousConvBnRelu(in_channels, branch_ch, kernel_size=3, dilation=2)
        self.b2 = _AtrousConvBnRelu(in_channels, branch_ch, kernel_size=3, dilation=4)
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-scale dilated context and global pooled features."""
        pooled = F.interpolate(self.pool_proj(x), size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.b0(x), self.b1(x), self.b2(x), pooled], dim=1))

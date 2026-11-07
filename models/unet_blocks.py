"""Building blocks for Attention U-Net."""

import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """Standard double convolution block: (Conv - BN - ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize double convolution block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through two conv-bn-relu layers."""
        return self.block(input_tensor)


class AdditiveAttentionGate(nn.Module):
    """Additive attention gate as proposed by Oktay et al."""

    def __init__(self, gate_channels: int, skip_channels: int, intermediate_channels: int) -> None:
        """Initialize attention projection layers."""
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate_tensor: torch.Tensor, skip_tensor: torch.Tensor) -> torch.Tensor:
        """Compute attention coefficient map and scale skip features."""
        gate_proj = self.gate_conv(gate_tensor)
        skip_proj = self.skip_conv(skip_tensor)
        attention_map = self.psi(self.relu(gate_proj + skip_proj))
        return skip_tensor * attention_map


class UpConvolutionAttentionBlock(nn.Module):
    """Upsampling followed by attention gating and double convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Initialize upsampling, attention gate, and convolution layers."""
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention_gate = AdditiveAttentionGate(
            gate_channels=out_channels,
            skip_channels=skip_channels,
            intermediate_channels=out_channels // 2,
        )
        self.conv_block = ConvolutionalBlock(out_channels + skip_channels, out_channels)

    def forward(self, decoder_tensor: torch.Tensor, skip_tensor: torch.Tensor) -> torch.Tensor:
        """Upsample, filter skip connection with attention gate, and fuse."""
        upsampled = self.upsample(decoder_tensor)
        gated_skip = self.attention_gate(gate_tensor=upsampled, skip_tensor=skip_tensor)
        concatenated = torch.cat([gated_skip, upsampled], dim=1)
        return self.conv_block(concatenated)


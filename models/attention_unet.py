"""Attention U-Net architecture with 5 downsampling and 5 upsampling stages."""

import torch
import torch.nn as nn
from mus_net.models.aspp_bottleneck import MultiScaleASPPBottleneck
from mus_net.models.unet_blocks import ConvolutionalBlock, UpConvolutionAttentionBlock


class AttentionUNet(nn.Module):
    """Configurable N-stage Attention U-Net with optional ASPP bottleneck."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_stages: int = 5,
        use_aspp: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize encoder, bottleneck (standard or ASPP), and decoder stages."""
        super().__init__()
        self.base_channels = base_channels
        channels = [base_channels * (2**i) for i in range(num_stages)]

        self.initial_conv = ConvolutionalBlock(in_channels, channels[0])
        self.encoder_convs = nn.ModuleList([
            ConvolutionalBlock(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = (
            MultiScaleASPPBottleneck(channels[-1], channels[-1] * 2, dropout_rate)
            if use_aspp else ConvolutionalBlock(channels[-1], channels[-1] * 2)
        )

        self.up_blocks = nn.ModuleList([
            UpConvolutionAttentionBlock(
                in_channels=channels[-1] * 2 if i == 0 else channels[len(channels) - i],
                skip_channels=channels[len(channels) - 1 - i],
                out_channels=channels[len(channels) - 1 - i],
            )
            for i in range(len(channels))
        ])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale representations and decode back to original resolution."""
        skips = [self.initial_conv(input_tensor)]
        current = skips[-1]

        for conv in self.encoder_convs:
            current = self.pool(current)
            current = conv(current)
            skips.append(current)

        bottleneck_features = self.bottleneck(self.pool(current))

        decoder_features = bottleneck_features
        for index, up_block in enumerate(self.up_blocks):
            skip = skips[len(skips) - 1 - index]
            decoder_features = up_block(decoder_features, skip)

        return decoder_features


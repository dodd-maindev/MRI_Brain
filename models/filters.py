"""Filter layers for Feature Filter Module (Sobel, Otsu, Gabor, Laplacian)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mus_net.models.filter_kernels import (
    create_sobel_kernels,
    create_gabor_kernel_bank,
    create_laplacian_kernel,
)
from mus_net.models.otsu_filter import OtsuThresholdFilter


class SobelFilter(nn.Module):
    """Depthwise Sobel gradient magnitude filter (Eq. 10 & 11)."""

    def __init__(self) -> None:
        """Initialize fixed Sobel kernels."""
        super().__init__()
        gx, gy = create_sobel_kernels()
        self.register_buffer("kernel_x", gx)
        self.register_buffer("kernel_y", gy)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply horizontal and vertical Sobel filtering depthwise."""
        c = input_tensor.shape[1]
        kx = self.kernel_x.repeat(c, 1, 1, 1)
        ky = self.kernel_y.repeat(c, 1, 1, 1)
        dx = F.conv2d(input_tensor, kx, padding=1, groups=c)
        dy = F.conv2d(input_tensor, ky, padding=1, groups=c)
        return torch.sqrt(dx**2 + dy**2 + 1e-7)


class GaborFilterBankModule(nn.Module):
    """Depthwise Gabor filter bank with channel projection (Eq. 19 & 20)."""

    def __init__(self, in_channels: int) -> None:
        """Initialize Gabor bank and 1x1 projection back to in_channels."""
        super().__init__()
        kernels = create_gabor_kernel_bank()
        self.num_kernels = kernels.shape[0]
        self.register_buffer("kernels", kernels)
        self.projector = nn.Conv2d(
            in_channels * self.num_kernels, in_channels, kernel_size=1, bias=False
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply Gabor bank depthwise and project back to channel dimension."""
        b, c, h, w = input_tensor.shape
        padded = F.pad(input_tensor, (7, 7, 7, 7), mode="reflect")
        filtered_list = []
        for i in range(self.num_kernels):
            k = self.kernels[i : i + 1].repeat(c, 1, 1, 1)
            filtered_list.append(F.conv2d(padded, k, groups=c))
        stacked = torch.cat(filtered_list, dim=1)
        return self.projector(stacked)


class LaplacianFilter(nn.Module):
    """Depthwise Laplacian second-derivative filter for ablation."""

    def __init__(self) -> None:
        """Initialize fixed Laplacian kernel."""
        super().__init__()
        self.register_buffer("kernel", create_laplacian_kernel())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply depthwise Laplacian operator."""
        c = input_tensor.shape[1]
        k = self.kernel.repeat(c, 1, 1, 1)
        return torch.abs(F.conv2d(input_tensor, k, padding=1, groups=c))


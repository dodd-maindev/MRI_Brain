"""Kernel generation routines for spatial filters."""

import math
from typing import List
import numpy as np
import torch


def create_sobel_kernels() -> tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal and vertical Sobel 3x3 kernels."""
    horizontal = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    vertical = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    return horizontal, vertical


def create_laplacian_kernel() -> torch.Tensor:
    """Return standard 3x3 discrete Laplacian kernel."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)
    return kernel


def create_gabor_kernel_bank(
    kernel_size: int = 15,
    frequencies: List[float] = None,
    orientations: List[float] = None,
    sigma: float = 3.0,
    gamma: float = 0.5,
    psi: float = 0.0,
) -> torch.Tensor:
    """Generate bank of real 2D Gabor filter kernels according to Eq. 19."""
    if frequencies is None:
        frequencies = [0.25]
    if orientations is None:
        orientations = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]

    radius = kernel_size // 2
    grid_y, grid_x = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1),
        indexing="ij",
    )

    kernels: List[torch.Tensor] = []
    for frequency in frequencies:
        wavelength = 1.0 / frequency
        for theta in orientations:
            x_prime = grid_x * math.cos(theta) + grid_y * math.sin(theta)
            y_prime = -grid_x * math.sin(theta) + grid_y * math.cos(theta)
            gaussian = np.exp(
                -(x_prime**2 + (gamma * y_prime) ** 2) / (2.0 * sigma**2)
            )
            sinusoid = np.cos(2.0 * math.pi * x_prime / wavelength + psi)
            gabor_kernel = gaussian * sinusoid
            gabor_kernel = gabor_kernel - np.mean(gabor_kernel)
            norm = np.linalg.norm(gabor_kernel)
            if norm > 1e-6:
                gabor_kernel = gabor_kernel / norm
            kernels.append(
                torch.tensor(gabor_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

    return torch.cat(kernels, dim=0)


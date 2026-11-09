"""VGG-style prior network for Conditional Variational Autoencoder (CVAE)."""

from typing import Tuple
import torch
import torch.nn as nn


class PriorNet(nn.Module):
    """Small VGG-style network estimating latent Gaussian parameters mu and sigma."""

    def __init__(self, in_channels: int = 1, latent_dimension: int = 6) -> None:
        """Initialize convolutional feature extractor and parameter heads."""
        super().__init__()
        self.latent_dimension = latent_dimension
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.mu_head = nn.Linear(128, latent_dimension)
        self.log_var_head = nn.Linear(128, latent_dimension)

    def forward(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output mean and log-variance vectors for input image."""
        flattened = self.features(input_tensor).flatten(1)
        mean_vector = self.mu_head(flattened)
        log_variance_vector = self.log_var_head(flattened)
        return mean_vector, log_variance_vector

    def sample(
        self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample latent vector z using reparameterization trick."""
        mean, log_variance = self.forward(input_tensor)
        if not self.training:
            return mean, mean, log_variance
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std)
        latent_sample = mean + epsilon * std
        return latent_sample, mean, log_variance


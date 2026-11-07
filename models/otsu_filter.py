"""Vectorized PyTorch implementation of Otsu thresholding."""

import torch
import torch.nn as nn


class OtsuThresholdFilter(nn.Module):
    """Batch-vectorized Otsu optimal thresholding module with STE support."""

    def __init__(self, number_of_bins: int = 256, straight_through: bool = True) -> None:
        """Initialize number of bins and gradient estimation flag."""
        super().__init__()
        self.number_of_bins = number_of_bins
        self.straight_through = straight_through

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute Otsu threshold per channel and return binary mask with gradients."""
        batch_size, channels, height, width = input_tensor.shape
        num_pixels = height * width
        epsilon = 1e-7

        min_val = input_tensor.amin(dim=(2, 3), keepdim=True)
        max_val = input_tensor.amax(dim=(2, 3), keepdim=True)
        normalized = (input_tensor - min_val) / (max_val - min_val + epsilon)
        scaled = normalized * (self.number_of_bins - 1)
        quantized = torch.clamp(scaled.long(), 0, self.number_of_bins - 1)

        channel_offsets = (
            torch.arange(batch_size * channels, device=input_tensor.device)
            * self.number_of_bins
        ).view(batch_size, channels, 1, 1)
        flat_indices = (quantized + channel_offsets).reshape(-1)

        histogram = torch.zeros(
            batch_size * channels * self.number_of_bins,
            device=input_tensor.device,
            dtype=torch.float32,
        )
        histogram.scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32)
        )
        hist_table = histogram.view(batch_size, channels, self.number_of_bins)

        probabilities = hist_table / float(num_pixels)
        omega1 = torch.cumsum(probabilities, dim=-1)
        omega2 = 1.0 - omega1

        bin_values = torch.arange(
            self.number_of_bins, device=input_tensor.device, dtype=torch.float32
        )
        cumulative_mean = torch.cumsum(probabilities * bin_values, dim=-1)
        total_mean = cumulative_mean[:, :, -1:]

        mu1 = cumulative_mean / (omega1 + epsilon)
        mu2 = (total_mean - cumulative_mean) / (omega2 + epsilon)
        sigma_b_squared = omega1 * omega2 * ((mu1 - mu2) ** 2)

        best_threshold_indices = torch.argmax(
            sigma_b_squared[:, :, :-1], dim=-1
        ).unsqueeze(-1).unsqueeze(-1)
        binary_output = (scaled > best_threshold_indices.float()).float()

        if self.straight_through:
            # ASSUMPTION: Straight-through estimator allows gradients to flow to d_i
            return binary_output + (input_tensor - input_tensor.detach())
        return binary_output.detach()


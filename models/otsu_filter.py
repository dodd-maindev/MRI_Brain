"""Vectorized PyTorch Otsu filter supporting Hard+STE and 100% Differentiable Kernel-Otsu."""

import torch
import torch.nn as nn


class OtsuThresholdFilter(nn.Module):
    """Otsu optimal thresholding with exact STE baseline or C-infinity Kernel-Otsu."""

    def __init__(
        self,
        number_of_bins: int = 256,
        straight_through: bool = True,
        soft_mode: bool = False,
        temperature: float = 0.05,
        kernel_bins: int = 32,
    ) -> None:
        """Initialize histogram bins, gradient mode, and Kernel-Otsu smoothing."""
        super().__init__()
        self.number_of_bins = number_of_bins
        self.straight_through = straight_through
        self.soft_mode = soft_mode
        self.temperature = temperature if temperature < 1.0 else temperature / 255.0
        self.kernel_bins = kernel_bins
        centers = torch.linspace(0.0, 1.0, kernel_bins)
        self.register_buffer("bin_centers", centers)

    def _forward_kernel_otsu(self, normalized: torch.Tensor) -> torch.Tensor:
        """Fully differentiable Gaussian-kernel histogram + Soft-Argmax inter-class variance."""
        b, c, h, w = normalized.shape
        x_flat = normalized.view(b, c, h * w, 1)
        centers = self.bin_centers.view(1, 1, 1, self.kernel_bins)
        sigma = 1.0 / float(self.kernel_bins)
        # Soft kernel density histogram p(k) differentiable w.r.t. input features
        soft_assign = torch.softmax(-((x_flat - centers) ** 2) / (2.0 * sigma * sigma), dim=-1)
        probs = soft_assign.mean(dim=2)

        eps = 1e-7
        omega1 = torch.cumsum(probs, dim=-1)
        omega2 = torch.clamp(1.0 - omega1, min=eps)
        bin_vals = self.bin_centers.view(1, 1, self.kernel_bins)
        cum_mean = torch.cumsum(probs * bin_vals, dim=-1)
        total_mean = cum_mean[:, :, -1:]
        mu1 = cum_mean / (omega1 + eps)
        mu2 = (total_mean - cum_mean) / omega2
        sigma_b_sq = omega1 * omega2 * ((mu1 - mu2) ** 2)

        # Differentiable Soft-Argmax optimal threshold t* in [0, 1]
        thresh_weights = torch.softmax(50.0 * sigma_b_sq[:, :, :-1], dim=-1)
        t_star = torch.sum(thresh_weights * bin_vals[:, :, :-1], dim=-1).unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid((normalized - t_star) / self.temperature)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute Otsu threshold per channel via Kernel-Otsu or Hard+STE."""
        b, c, h, w = input_tensor.shape
        eps = 1e-7
        min_v = input_tensor.amin(dim=(2, 3), keepdim=True)
        max_v = input_tensor.amax(dim=(2, 3), keepdim=True)
        normalized = (input_tensor - min_v) / (max_v - min_v + eps)
        if self.soft_mode:
            return self._forward_kernel_otsu(normalized)

        scaled = normalized * (self.number_of_bins - 1)
        quantized = torch.clamp(scaled.long(), 0, self.number_of_bins - 1)
        offsets = (torch.arange(b * c, device=input_tensor.device) * self.number_of_bins).view(b, c, 1, 1)
        hist = torch.zeros(b * c * self.number_of_bins, device=input_tensor.device, dtype=torch.float32)
        flat_idx = (quantized + offsets).reshape(-1)
        hist.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        probs = hist.view(b, c, self.number_of_bins) / float(h * w)
        omega1 = torch.cumsum(probs, dim=-1)
        omega2 = 1.0 - omega1
        bins = torch.arange(self.number_of_bins, device=input_tensor.device, dtype=torch.float32)
        cum_mean = torch.cumsum(probs * bins, dim=-1)
        mu1, mu2 = cum_mean / (omega1 + eps), (cum_mean[:, :, -1:] - cum_mean) / (omega2 + eps)
        sigma_b_sq = omega1 * omega2 * ((mu1 - mu2) ** 2)
        best_t = torch.argmax(sigma_b_sq[:, :, :-1], dim=-1).unsqueeze(-1).unsqueeze(-1)
        binary = (scaled > best_t.float()).float()
        return binary + (input_tensor - input_tensor.detach()) if self.straight_through else binary.detach()

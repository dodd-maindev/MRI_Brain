"""Image-conditioned Adaptive Residual Windowing (AdaWindow) for CT HU inputs."""

from typing import Dict
import torch
import torch.nn as nn


class AdaptiveMultiWindowModule(nn.Module):
    """Learns slice-adaptive residual WL/WW offsets around radiological priors."""

    def __init__(self, raw_min: float = -1000.0, raw_max: float = 400.0) -> None:
        """Initialize residual window predictors anchored at Lung and Medias priors."""
        super().__init__()
        self.raw_min, self.raw_scale = raw_min, raw_max - raw_min
        self.base_offsets = nn.Parameter(torch.zeros(4))
        self.context_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 4),
        )
        nn.init.zeros_(self.context_net[-1].weight)
        nn.init.zeros_(self.context_net[-1].bias)

    @staticmethod
    def _apply_window(hu: torch.Tensor, wl: torch.Tensor, ww: torch.Tensor) -> torch.Tensor:
        """Apply differentiable linear CT windowing clamped to [0, 1]."""
        low = wl - 0.5 * ww
        return torch.clamp((hu - low) / (ww + 1e-6), 0.0, 1.0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate slice-adaptive lung and mediastinal windows across all input channels."""
        raw = inputs["raw"]
        hu = raw * self.raw_scale + self.raw_min
        center_raw = raw[:, 1:2] if raw.shape[1] >= 3 else raw[:, 0:1]
        delta = self.base_offsets.view(1, 4) + self.context_net(center_raw)
        d = delta.unsqueeze(-1).unsqueeze(-1)

        wl_lung = -600.0 + 120.0 * torch.tanh(d[:, 0:1])
        ww_lung = 1500.0 + 250.0 * torch.tanh(d[:, 1:2])
        wl_med = 40.0 + 80.0 * torch.tanh(d[:, 2:3])
        ww_med = 400.0 + 150.0 * torch.tanh(d[:, 3:4])

        return {
            "raw": raw,
            "lung": self._apply_window(hu, wl_lung, ww_lung),
            "medias": self._apply_window(hu, wl_med, ww_med),
        }

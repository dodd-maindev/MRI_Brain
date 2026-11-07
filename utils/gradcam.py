"""Grad-CAM implementation for visualizing target convolutional activations."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCamExplainer:
    """Computes Gradient-weighted Class Activation Mapping for CNN layers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Register forward and backward hooks on target layer."""
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(
        self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor
    ) -> None:
        """Capture forward activations."""
        self.activations = output_tensor.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        model_output: torch.Tensor,
        spatial_size: tuple[int, int] = (128, 128),
    ) -> torch.Tensor:
        """Backpropagate score and compute localized Grad-CAM heatmaps."""
        self.model.zero_grad()
        score = model_output.sum()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to collect gradients or activations for Grad-CAM.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(weighted_activations)

        cam_resized = F.interpolate(
            cam, size=spatial_size, mode="bilinear", align_corners=False
        )
        cam_min = cam_resized.amin(dim=(2, 3), keepdim=True)
        cam_max = cam_resized.amax(dim=(2, 3), keepdim=True)
        epsilon = 1e-7
        return (cam_resized - cam_min) / (cam_max - cam_min + epsilon)


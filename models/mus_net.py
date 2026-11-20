"""Multi-Window Uncertainty-Guided Network (MUS-Net) complete architecture."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from mus_net.models.prob_attention_unet import ProbabilisticAttentionUNet
from mus_net.models.uim import UncertaintyIndicationModule, compute_multi_window_mask
from mus_net.models.ffm import FeatureFilterBranch
from mus_net.models.adaptive_window import AdaptiveMultiWindowModule
from mus_net.models.uncertainty_gate import CrossWindowUncertaintyGate
from mus_net.models.unet_blocks import ConvolutionalBlock


class MultiWindowSegmentationNetwork(nn.Module):
    """Represents MUS-Net combining FEM branches, UIM, and FFM for 2D lung nodule CT."""

    def __init__(
        self,
        base_channels: int = 32,
        latent_dimension: int = 6,
        beta: float = 1.0,
        attention_downsample_ratio: int = 4,
        lung_filter: str = "sobel",
        medias_filter: str = "otsu",
        raw_filter: str = "gabor",
        active_branches: Optional[List[str]] = None,
        use_uim: bool = True,
        use_ffm: bool = True,
        use_uncertainty_gate: bool = False,
        use_adaptive_window: bool = False,
        otsu_soft: bool = False,
        otsu_temperature: float = 8.0,
    ) -> None:
        """Initialize all branches, filters, optional AdaWindow/uncertainty gate, and fusion head."""
        super().__init__()
        self.active_branches = active_branches or ["lung", "medias", "raw"]
        self.use_uim, self.use_ffm, self.use_uncertainty_gate = use_uim, use_ffm, use_uncertainty_gate
        self.adaptive_window = AdaptiveMultiWindowModule() if use_adaptive_window else None
        c_fem = base_channels + latent_dimension
        self.fem = nn.ModuleDict({b: ProbabilisticAttentionUNet(1, base_channels, latent_dimension) for b in ["lung", "medias", "raw"]})
        self.uim = nn.ModuleDict({b: UncertaintyIndicationModule(c_fem, base_channels) for b in ["lung", "medias", "raw"]})
        filter_types = {"lung": lung_filter, "medias": medias_filter, "raw": raw_filter}
        self.ffm = nn.ModuleDict({
            b: FeatureFilterBranch(base_channels, filter_types[b], beta, attention_downsample_ratio, otsu_soft, otsu_temperature)
            for b in ["lung", "medias", "raw"]
        })
        self.uncertainty_gate = CrossWindowUncertaintyGate(base_channels, self.active_branches) if use_uncertainty_gate else None
        num_active = len(self.active_branches)
        fusion_in_channels = num_active * base_channels + num_active * c_fem
        self.fusion_head = nn.Sequential(
            ConvolutionalBlock(fusion_in_channels, base_channels * 2),
            ConvolutionalBlock(base_channels * 2, base_channels),
            ConvolutionalBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute full forward pass across multi-window inputs."""
        if self.adaptive_window is not None:
            inputs = self.adaptive_window(inputs)
        f_maps, uim_feats, masks, latent_params = {}, {}, {}, {}

        for b in self.active_branches:
            f_maps[b], mu, log_var = self.fem[b](inputs[b])
            latent_params[b] = (mu, log_var)
            if self.use_uim:
                uim_feats[b], masks[b] = self.uim[b](f_maps[b])
            else:
                uim_feats[b], masks[b] = f_maps[b][:, : self.ffm[b].attention.in_channels], None

        y_features = [
            self.ffm[b](uim_feats[b]) if self.use_ffm else uim_feats[b]
            for b in self.active_branches
        ]
        f_features = [f_maps[b] for b in self.active_branches]

        # Fusion across active branches (Eq. 22 or MWM-guided dynamic uncertainty gating)
        if self.uncertainty_gate is not None and self.use_uim:
            fused_tensor, window_weights, _ = self.uncertainty_gate(y_features, f_features, masks)
        else:
            fused_tensor, window_weights = torch.cat(y_features + f_features, dim=1), None
        final_prediction = self.fusion_head(fused_tensor)

        mwm = None
        if "lung" in masks and "medias" in masks and masks["lung"] is not None:
            mwm = compute_multi_window_mask(masks["lung"], masks["medias"])

        return {
            "final_prediction": final_prediction,
            "branch_masks": masks,
            "latent_parameters": latent_params,
            "multi_window_mask": mwm,
            "window_weights": window_weights,
        }


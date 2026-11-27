"""Multi-Window Uncertainty-Guided Network (MUS-Net / AMU-Net / EDL-MUS-Net)."""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from mus_net.models.adaptive_window import AdaptiveMultiWindowModule
from mus_net.models.evidential_fusion import (
    DempsterShaferWindowFusion, EvidentialProjectionHead, combine_two_opinions, opinion_to_beta_and_prob,
)
from mus_net.models.ffm import FeatureFilterBranch
from mus_net.models.prob_attention_unet import ProbabilisticAttentionUNet
from mus_net.models.uim import UncertaintyIndicationModule, compute_multi_window_mask
from mus_net.models.uncertainty_gate import CrossWindowUncertaintyGate
from mus_net.models.unet_blocks import ConvolutionalBlock


class MultiWindowSegmentationNetwork(nn.Module):
    """Unified multi-window segmentation network supporting MUS-Net, AMU-Net, and EDL-MUS-Net."""

    def __init__(
        self, base_channels: int = 32, latent_dimension: int = 6, beta: float = 1.0,
        attention_downsample_ratio: int = 4, lung_filter: str = "sobel", medias_filter: str = "otsu",
        raw_filter: str = "gabor", active_branches: Optional[List[str]] = None,
        use_uim: bool = True, use_ffm: bool = True, use_uncertainty_gate: bool = False,
        use_adaptive_window: bool = False, otsu_soft: bool = False, otsu_temperature: float = 0.05,
        unet_stages: int = 5, use_aspp_bottleneck: bool = False, dropout_rate: float = 0.0,
        in_channels: int = 1, use_evidential: bool = False,
    ) -> None:
        """Initialize FEM branches, UIM, FFM, optional AdaWindow/EDL-MWF, and fusion head."""
        super().__init__()
        self.active_branches = active_branches or ["lung", "medias", "raw"]
        self.use_uim, self.use_ffm, self.use_evidential = use_uim, use_ffm, use_evidential
        self.adaptive_window = AdaptiveMultiWindowModule() if use_adaptive_window else None
        c_fem = base_channels + latent_dimension
        self.fem = nn.ModuleDict({
            b: ProbabilisticAttentionUNet(in_channels, base_channels, latent_dimension, unet_stages, use_aspp_bottleneck, dropout_rate)
            for b in ["lung", "medias", "raw"]
        })
        self.uim = nn.ModuleDict({b: UncertaintyIndicationModule(c_fem, base_channels) for b in ["lung", "medias", "raw"]})
        ft = {"lung": lung_filter, "medias": medias_filter, "raw": raw_filter}
        self.ffm = nn.ModuleDict({
            b: FeatureFilterBranch(base_channels, ft[b], beta, attention_downsample_ratio, otsu_soft, otsu_temperature)
            for b in ["lung", "medias", "raw"]
        })
        self.evidential_heads = nn.ModuleDict({b: EvidentialProjectionHead(base_channels) for b in ["lung", "medias", "raw"]}) if use_evidential else None
        self.ds_fusion = DempsterShaferWindowFusion(self.active_branches) if use_evidential else None
        self.uncertainty_gate = CrossWindowUncertaintyGate(base_channels, self.active_branches) if (use_uncertainty_gate and not use_evidential) else None

        fusion_in = len(self.active_branches) * (base_channels + c_fem)
        head_out: List[nn.Module] = (
            [EvidentialProjectionHead(base_channels)] if use_evidential
            else [nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid()]
        )
        self.fusion_head = nn.Sequential(
            ConvolutionalBlock(fusion_in, base_channels * 2),
            ConvolutionalBlock(base_channels * 2, base_channels),
            ConvolutionalBlock(base_channels, base_channels),
            *head_out,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute multi-window forward pass with optional Dempster-Shafer evidential fusion."""
        if self.adaptive_window is not None:
            inputs = self.adaptive_window(inputs)
        f_maps, uim_feats, masks, latent_params, opinions, branch_ev = {}, {}, {}, {}, {}, {}
        for b in self.active_branches:
            f_maps[b], mu, log_var = self.fem[b](inputs[b])
            latent_params[b] = (mu, log_var)
            if self.use_uim:
                uim_feats[b], masks[b] = self.uim[b](f_maps[b])
                if self.evidential_heads is not None:
                    a_b, bt_b, op_b = self.evidential_heads[b](uim_feats[b])
                    branch_ev[b], opinions[b], masks[b] = (a_b, bt_b), op_b, torch.clamp(op_b[0] + 0.5 * op_b[2], 1e-6, 1.0 - 1e-6)
            else:
                uim_feats[b], masks[b] = f_maps[b][:, : self.ffm[b].attention.in_channels], None

        y_feats = [self.ffm[b](uim_feats[b]) if self.use_ffm else uim_feats[b] for b in self.active_branches]
        f_feats = [f_maps[b] for b in self.active_branches]
        final_ev, epi_u, conflict_lm, window_weights = None, None, None, None

        if self.ds_fusion is not None and self.use_uim:
            fused, op_mw, window_weights, conflict_lm = self.ds_fusion(y_feats, f_feats, opinions)
            _, _, op_head = self.fusion_head(fused)
            op_final, _ = combine_two_opinions(op_head, op_mw)
            a_f, bt_f, final_pred = opinion_to_beta_and_prob(op_final)
            final_ev, epi_u = (a_f, bt_f), op_final[2]
        elif self.uncertainty_gate is not None and self.use_uim:
            fused, window_weights, _ = self.uncertainty_gate(y_feats, f_feats, masks)
            final_pred = self.fusion_head(fused)
        else:
            final_pred = self.fusion_head(torch.cat(y_feats + f_feats, dim=1))

        mwm = compute_multi_window_mask(masks["lung"], masks["medias"]) if (
            masks.get("lung") is not None and masks.get("medias") is not None
        ) else None
        return {
            "final_prediction": final_pred, "branch_masks": masks, "latent_parameters": latent_params,
            "multi_window_mask": mwm, "window_weights": window_weights, "final_evidence": final_ev,
            "branch_evidence": branch_ev if branch_ev else None, "epistemic_uncertainty": epi_u, "conflict_map": conflict_lm,
        }

"""Unit tests verifying tensor shapes across all MUS-Net modules."""

import pytest
import torch
from mus_net.models.attention_unet import AttentionUNet
from mus_net.models.prior_net import PriorNet
from mus_net.models.prob_attention_unet import ProbabilisticAttentionUNet
from mus_net.models.uim import UncertaintyIndicationModule, compute_multi_window_mask
from mus_net.models.ffm import FeatureFilterBranch
from mus_net.models.mus_net import MultiWindowSegmentationNetwork


@pytest.fixture
def batch_tensor() -> torch.Tensor:
    """Fixture producing standard CT slice batch [2, 1, 128, 128]."""
    return torch.randn(2, 1, 128, 128)


def test_attention_unet_shape(batch_tensor: torch.Tensor) -> None:
    """Verify Attention U-Net outputs [B, base_channels, H, W]."""
    model = AttentionUNet(in_channels=1, base_channels=32)
    output = model(batch_tensor)
    assert output.shape == (2, 32, 128, 128)


def test_prior_net_shape(batch_tensor: torch.Tensor) -> None:
    """Verify PriorNet produces latent parameters of dimension N."""
    model = PriorNet(in_channels=1, latent_dimension=6)
    mu, log_var = model(batch_tensor)
    sample, _, _ = model.sample(batch_tensor)
    assert mu.shape == (2, 6)
    assert log_var.shape == (2, 6)
    assert sample.shape == (2, 6)


def test_probabilistic_attention_unet_shape(batch_tensor: torch.Tensor) -> None:
    """Verify FEM branch concatenates deterministic and latent representations."""
    model = ProbabilisticAttentionUNet(in_channels=1, base_channels=32, latent_dimension=6)
    f_i, mu, log_var = model(batch_tensor)
    assert f_i.shape == (2, 38, 128, 128)
    assert mu.shape == (2, 6)


def test_uncertainty_indication_module_shape() -> None:
    """Verify UIM outputs intermediate feature map F_i and preliminary mask S_i."""
    f_i = torch.randn(2, 38, 128, 128)
    uim = UncertaintyIndicationModule(in_channels=38, intermediate_channels=32)
    f_out, s_out = uim(f_i)
    assert f_out.shape == (2, 32, 128, 128)
    assert s_out.shape == (2, 1, 128, 128)
    mwm = compute_multi_window_mask(s_out, s_out)
    assert mwm.shape == (2, 3, 128, 128)


@pytest.mark.parametrize("filter_name", ["sobel", "otsu", "gabor", "laplacian"])
def test_feature_filter_branch_shape(filter_name: str) -> None:
    """Verify all FFM filter variants preserve channel and spatial dimensions."""
    f_i = torch.randn(2, 32, 128, 128)
    ffm = FeatureFilterBranch(channels=32, filter_type=filter_name, downsample_ratio=4)
    y_i = ffm(f_i)
    assert y_i.shape == (2, 32, 128, 128)


def test_full_mus_net_end_to_end_shape() -> None:
    """Verify full MUS-Net maps 3x [B, 1, 128, 128] to final segmentation [B, 1, 128, 128]."""
    inputs = {k: torch.randn(2, 1, 128, 128) for k in ["raw", "lung", "medias"]}
    model = MultiWindowSegmentationNetwork(base_channels=32, latent_dimension=6)
    outputs = model(inputs)

    assert outputs["final_prediction"].shape == (2, 1, 128, 128)
    assert outputs["branch_masks"]["lung"].shape == (2, 1, 128, 128)
    assert outputs["branch_masks"]["medias"].shape == (2, 1, 128, 128)
    assert outputs["branch_masks"]["raw"].shape == (2, 1, 128, 128)
    assert outputs["multi_window_mask"].shape == (2, 3, 128, 128)


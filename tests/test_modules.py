"""Unit tests for windowing, metrics, loss functions, and data splitting."""

import numpy as np
import torch
from mus_net.data.patient_splitter import PatientWiseCrossValidationSplitter
from mus_net.data.windowing import ComputedTomographyWindowTransformer
from mus_net.losses import MultiWindowSegmentationLoss
from mus_net.evidential_loss import compute_evidential_beta_loss
from mus_net.models.evidential_fusion import combine_two_opinions
from mus_net.models.otsu_filter import OtsuThresholdFilter
from mus_net.metrics import (
    compute_dice_coefficient,
    compute_intersection_over_union,
    compute_normalized_surface_distance,
)


def test_windowing_transformer_ranges() -> None:
    """Verify that CT windowing produces outputs strictly bounded in [0, 1]."""
    transformer = ComputedTomographyWindowTransformer()
    hu_input = np.linspace(-1200, 600, 1000).reshape((1, 10, 100))
    result = transformer.transform(hu_input)

    for key in ["raw", "lung", "medias"]:
        assert result[key].min() >= 0.0
        assert result[key].max() <= 1.0


def test_patient_cross_validation_splitting_leakage() -> None:
    """Ensure no patient overlap exists between train and validation folds."""
    patients = ["P1", "P1", "P2", "P3", "P3", "P4", "P5", "P6", "P7", "P8"]
    splitter = PatientWiseCrossValidationSplitter(number_of_folds=5, random_seed=42)
    folds = splitter.split(patients)

    assert len(folds) == 5
    for fold in folds:
        train_p = {patients[i] for i in fold["train_indices"]}
        val_p = {patients[i] for i in fold["validation_indices"]}
        assert len(train_p.intersection(val_p)) == 0


def test_segmentation_metrics_perfect_and_empty() -> None:
    """Verify metric boundaries on synthetic masks."""
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[20:40, 20:40] = 1
    pred = gt.copy()

    assert compute_dice_coefficient(gt, pred) == 1.0
    assert compute_intersection_over_union(gt, pred) == 1.0
    assert compute_normalized_surface_distance(gt, pred, tau=1.0) == 1.0

    empty = np.zeros((64, 64), dtype=np.uint8)
    assert compute_dice_coefficient(gt, empty) == 0.0
    assert compute_intersection_over_union(gt, empty) == 0.0
    assert compute_normalized_surface_distance(gt, empty, tau=1.0) == 0.0


def test_loss_function_computation() -> None:
    """Verify that multi-window segmentation loss returns positive scalar with gradients."""
    criterion = MultiWindowSegmentationLoss(alpha_1=1.0, alpha_2=0.5, use_posterior_kl=True)
    b, c, h, w = 2, 1, 32, 32
    target = torch.ones((b, c, h, w), dtype=torch.float32)

    leaf_pred = torch.rand(b, c, h, w, requires_grad=True)
    outputs = {
        "final_prediction": leaf_pred,
        "branch_masks": {
            "lung": torch.rand(b, c, h, w, requires_grad=True),
            "medias": torch.rand(b, c, h, w, requires_grad=True),
            "raw": torch.rand(b, c, h, w, requires_grad=True),
        },
        "latent_parameters": {
            "lung": (torch.randn(b, 6, requires_grad=True), torch.randn(b, 6, requires_grad=True)),
        },
    }
    loss, details = criterion(outputs, target)
    assert loss.item() > 0.0
    assert "total_loss" in details
    loss.backward()
    assert leaf_pred.grad is not None


def test_dempster_shafer_propositions_and_kernel_otsu_grad() -> None:
    """Verify DS blind-window invariance (Prop. 1), uncertainty drop (Prop. 2), and Kernel-Otsu grad."""
    op1 = (torch.tensor([[[[0.6]]]]), torch.tensor([[[[0.2]]]]), torch.tensor([[[[0.2]]]]))
    blind = (torch.zeros_like(op1[0]), torch.zeros_like(op1[0]), torch.ones_like(op1[0]))
    fused_blind, conflict = combine_two_opinions(op1, blind)
    assert torch.allclose(conflict, torch.zeros_like(conflict))
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(fused_blind, op1))
    op2 = (torch.tensor([[[[0.5]]]]), torch.tensor([[[[0.1]]]]), torch.tensor([[[[0.4]]]]))
    assert float(combine_two_opinions(op1, op2)[0][2]) < min(float(op1[2]), float(op2[2]))
    x = torch.randn(2, 4, 16, 16, requires_grad=True)
    OtsuThresholdFilter(soft_mode=True)(x).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0
    edl = compute_evidential_beta_loss(torch.full((2, 1, 8, 8), 3.0, requires_grad=True), torch.full((2, 1, 8, 8), 2.0), torch.ones(2, 1, 8, 8))
    assert float(edl) > 0.0

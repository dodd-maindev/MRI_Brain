"""Q1 geometric and uncertainty calibration metrics: HD95, ECE, and AURC."""

from typing import Dict
import numpy as np
import scipy.ndimage as ndi


def compute_slice_hd95(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff Distance (in pixels) for one 2D slice."""
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    if not gt_b.any() and not pred_b.any():
        return 0.0
    if not gt_b.any() or not pred_b.any():
        return float(np.hypot(*gt_b.shape))
    struct = ndi.generate_binary_structure(2, 1)
    b_gt = gt_b ^ ndi.binary_erosion(gt_b, structure=struct)
    b_pred = pred_b ^ ndi.binary_erosion(pred_b, structure=struct)
    d_gt = ndi.distance_transform_edt(~b_gt)[b_pred]
    d_pred = ndi.distance_transform_edt(~b_pred)[b_gt]
    return float(np.percentile(np.concatenate([d_gt, d_pred]), 95))


def compute_calibration_and_aurc(
    probs: np.ndarray, targets: np.ndarray, num_bins: int = 10
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Area Under Risk-Coverage Curve."""
    p = np.clip(probs.reshape(-1), 1e-6, 1.0 - 1e-6)
    y = (targets.reshape(-1) >= 0.5).astype(np.float32)
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(np.float32) == y).astype(np.float32)

    ece = 0.0
    bins = np.linspace(0.5, 1.0, num_bins + 1)
    for i in range(num_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1]) if i > 0 else (conf >= bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            ece += (mask.mean()) * abs(correct[mask].mean() - conf[mask].mean())

    # Subsample up to 200k pixels for fast deterministic AURC calculation
    if len(p) > 200_000:
        step = len(p) // 200_000
        p, correct = p[::step], correct[::step]
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
    order = np.argsort(entropy)
    errors = 1.0 - correct[order]
    cum_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    aurc = float(np.mean(cum_risk))
    return {"hd95": 0.0, "ece": float(ece), "aurc": aurc}

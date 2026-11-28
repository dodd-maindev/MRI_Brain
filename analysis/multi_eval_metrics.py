"""Metric helpers for multi-annotator evaluation: AUROC, partial corr, Brier/ECE(y_bar), LOO."""

from typing import Dict
import numpy as np
import scipy.ndimage as ndi
from scipy import stats
from sklearn.metrics import roc_auc_score


def safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC safely, returning NaN if labels contain only one class."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Compute partial correlation between x and y controlling for z via OLS residuals."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(stats.spearmanr(rx, ry).statistic)


def compute_soft_calibration(probs: np.ndarray, y_bar: np.ndarray, num_bins: int = 10) -> Dict[str, float]:
    """Compute Brier score and 10-bin ECE against soft consensus target y_bar."""
    p = np.clip(probs.reshape(-1), 1e-6, 1.0 - 1e-6)
    yb = np.clip(y_bar.reshape(-1), 0.0, 1.0)
    brier = float(np.mean((p - yb) ** 2))
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        m = (p > bins[i]) & (p <= bins[i + 1]) if i > 0 else (p >= bins[i]) & (p <= bins[i + 1])
        if m.any():
            ece += float(m.mean()) * abs(float(yb[m].mean()) - float(p[m].mean()))
    return {"brier": brier, "ece": float(ece)}


def compute_union_roi_mask(ann_masks: np.ndarray, num_anns: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    """Build dilated union mask across present annotators for pixel uncertainty evaluation."""
    struct = ndi.generate_binary_structure(2, 1)
    roi = np.zeros((ann_masks.shape[0], ann_masks.shape[2], ann_masks.shape[3]), dtype=bool)
    for i in range(ann_masks.shape[0]):
        n = min(int(num_anns[i]), 4)
        u = ann_masks[i, :n].sum(axis=0) > 0
        roi[i] = ndi.binary_dilation(u, structure=struct, iterations=dilate_px) if u.any() else True
    return roi


def compute_loo_dsc(preds_bin: np.ndarray, ann_masks: np.ndarray, num_anns: np.ndarray) -> float:
    """Compute leave-one-out Global DSC (%) on slices with n >= 3."""
    inter, denom = 0, 0
    for i in range(len(preds_bin)):
        n = min(int(num_anns[i]), 4)
        if n < 3:
            continue
        stack = ann_masks[i, :n]
        tot = stack.sum(axis=0)
        for j in range(n):
            cons = ((tot - stack[j]) >= 0.5 * (n - 1)).astype(np.uint8)
            inter += int(np.logical_and(preds_bin[i], cons).sum())
            denom += int(preds_bin[i].sum()) + int(cons.sum())
    return 100.0 if denom == 0 else 200.0 * inter / denom

"""Segmentation evaluation metrics: DSC, IoU, and NSD."""

from typing import Dict, List
import numpy as np
import scipy.ndimage as ndi
import torch


def compute_dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient (DSC) (Eq. 25)."""
    gt_bool, pred_bool = ground_truth.astype(bool), prediction.astype(bool)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    total_area = gt_bool.sum() + pred_bool.sum()
    return 1.0 if total_area == 0 else float(2.0 * intersection / total_area)


def compute_intersection_over_union(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) (Eq. 26)."""
    gt_bool, pred_bool = ground_truth.astype(bool), prediction.astype(bool)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    return 1.0 if union == 0 else float(intersection / union)


def compute_normalized_surface_distance(
    ground_truth: np.ndarray, prediction: np.ndarray, tau_threshold: float = 1.0
) -> float:
    """Compute Normalized Surface Distance (NSD) (Eq. 27)."""
    gt_bool, pred_bool = ground_truth.astype(bool), prediction.astype(bool)
    struct = ndi.generate_binary_structure(2, 1)
    b_gt = gt_bool ^ ndi.binary_erosion(gt_bool, structure=struct)
    b_pred = pred_bool ^ ndi.binary_erosion(pred_bool, structure=struct)
    n_gt, n_pred = int(b_gt.sum()), int(b_pred.sum())

    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    d_pred, d_gt = ndi.distance_transform_edt(~b_pred), ndi.distance_transform_edt(~b_gt)
    match_gt = np.logical_and(b_gt, d_pred <= tau_threshold).sum()
    match_pred = np.logical_and(b_pred, d_gt <= tau_threshold).sum()
    return float((match_gt + match_pred) / (n_gt + n_pred))


class SegmentationMetricAccumulator:
    """Accumulates batch predictions to compute both Global/Volumetric and Macro metrics."""

    def __init__(self, tau: float = 1.0, threshold: float = 0.5) -> None:
        """Initialize counters and tolerance."""
        self.tau, self.threshold = tau, threshold
        self.tot_inter, self.tot_gt, self.tot_pred = 0, 0, 0
        self.slice_dscs, self.slice_ious, self.slice_nsds = [], [], []

    def update(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> None:
        """Ingest batch and accumulate pixel overlap statistics."""
        gt_np = (ground_truth >= 0.5).detach().cpu().numpy().astype(np.uint8)
        pred_np = (prediction >= self.threshold).detach().cpu().numpy().astype(np.uint8)

        for i in range(gt_np.shape[0]):
            g, p = gt_np[i, 0], pred_np[i, 0]
            inter = np.logical_and(g, p).sum()
            self.tot_inter += inter
            self.tot_gt += g.sum()
            self.tot_pred += p.sum()
            self.slice_dscs.append(compute_dice_coefficient(g, p))
            self.slice_ious.append(compute_intersection_over_union(g, p))
            self.slice_nsds.append(compute_normalized_surface_distance(g, p, self.tau))

    def compute(self) -> Dict[str, float]:
        """Return Global Volumetric (Eq. 25-27 in R^3) and Slice-wise Macro scores."""
        denom = self.tot_gt + self.tot_pred
        global_dsc = 1.0 if denom == 0 else float(2.0 * self.tot_inter / denom)
        global_union = self.tot_gt + self.tot_pred - self.tot_inter
        global_iou = 1.0 if global_union == 0 else float(self.tot_inter / global_union)
        return {
            "global_dsc": global_dsc,
            "global_iou": global_iou,
            "macro_dsc": float(np.mean(self.slice_dscs)),
            "macro_iou": float(np.mean(self.slice_ious)),
            "nsd": float(np.mean(self.slice_nsds)),
        }


def evaluate_segmentation_batch(
    ground_truth: torch.Tensor, prediction: torch.Tensor, tau_threshold: float = 1.0, threshold: float = 0.5
) -> Dict[str, float]:
    """Compatibility wrapper returning batch mean scores."""
    acc = SegmentationMetricAccumulator(tau=tau_threshold, threshold=threshold)
    acc.update(ground_truth, prediction)
    res = acc.compute()
    return {"dsc": res["global_dsc"], "macro_dsc": res["macro_dsc"], "iou": res["global_iou"], "nsd": res["nsd"]}

"""Segmentation evaluation metrics: DSC, IoU, and NSD."""

from typing import Dict, List
import numpy as np
import scipy.ndimage as ndi
import torch


def compute_dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient (DSC) (Eq. 25)."""
    gt_b, pred_b = ground_truth.astype(bool), prediction.astype(bool)
    tot = gt_b.sum() + pred_b.sum()
    return 1.0 if tot == 0 else float(2.0 * np.logical_and(gt_b, pred_b).sum() / tot)


def compute_intersection_over_union(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) (Eq. 26)."""
    gt_b, pred_b = ground_truth.astype(bool), prediction.astype(bool)
    union = np.logical_or(gt_b, pred_b).sum()
    return 1.0 if union == 0 else float(np.logical_and(gt_b, pred_b).sum() / union)


def compute_normalized_surface_distance(gt: np.ndarray, pred: np.ndarray, tau: float = 1.0) -> float:
    """Compute Normalized Surface Distance (NSD) (Eq. 27)."""
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    struct = ndi.generate_binary_structure(2, 1)
    b_gt, b_pred = gt_b ^ ndi.binary_erosion(gt_b, struct), pred_b ^ ndi.binary_erosion(pred_b, struct)
    n_gt, n_pred = int(b_gt.sum()), int(b_pred.sum())
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0
    m_gt = np.logical_and(b_gt, ndi.distance_transform_edt(~b_pred) <= tau).sum()
    m_pred = np.logical_and(b_pred, ndi.distance_transform_edt(~b_gt) <= tau).sum()
    return float((m_gt + m_pred) / (n_gt + n_pred))


class SegmentationMetricAccumulator:
    """Accumulates batch predictions to compute Global Volumetric (Eq. 25-27) and Macro scores."""

    def __init__(self, tau: float = 1.0, threshold: float = 0.5) -> None:
        """Initialize counters and tolerance."""
        self.tau, self.threshold = tau, threshold
        self.tot_inter, self.tot_gt, self.tot_pred = 0, 0, 0
        self.tot_surf_match, self.tot_surf_pixels = 0, 0
        self.slice_dscs, self.slice_ious, self.slice_nsds = [], [], []

    def update(self, ground_truth: torch.Tensor, prediction: torch.Tensor) -> None:
        """Ingest batch and accumulate pixel overlap and surface statistics."""
        gt_np = (ground_truth >= 0.5).detach().cpu().numpy().astype(np.uint8)
        pred_np = (prediction >= self.threshold).detach().cpu().numpy().astype(np.uint8)
        struct = ndi.generate_binary_structure(2, 1)

        for i in range(gt_np.shape[0]):
            g, p = gt_np[i, 0], pred_np[i, 0]
            inter = np.logical_and(g, p).sum()
            self.tot_inter += inter
            self.tot_gt += g.sum()
            self.tot_pred += p.sum()
            self.slice_dscs.append(compute_dice_coefficient(g, p))
            self.slice_ious.append(compute_intersection_over_union(g, p))

            b_gt = (g.astype(bool)) ^ ndi.binary_erosion(g.astype(bool), structure=struct)
            b_pred = (p.astype(bool)) ^ ndi.binary_erosion(p.astype(bool), structure=struct)
            n_gt, n_pred = int(b_gt.sum()), int(b_pred.sum())
            if n_gt > 0 or n_pred > 0:
                d_pred, d_gt = ndi.distance_transform_edt(~b_pred), ndi.distance_transform_edt(~b_gt)
                m_gt = np.logical_and(b_gt, d_pred <= self.tau).sum()
                m_pred = np.logical_and(b_pred, d_gt <= self.tau).sum()
                self.tot_surf_match += int(m_gt + m_pred)
                self.tot_surf_pixels += int(n_gt + n_pred)
                self.slice_nsds.append(float((m_gt + m_pred) / (n_gt + n_pred)))
            else:
                self.slice_nsds.append(1.0)

    def compute(self) -> Dict[str, float]:
        """Return Global Volumetric (Eq. 25-27 in R^3) and Slice-wise Macro scores."""
        denom = self.tot_gt + self.tot_pred
        g_dsc = 1.0 if denom == 0 else float(2.0 * self.tot_inter / denom)
        g_union = self.tot_gt + self.tot_pred - self.tot_inter
        g_iou = 1.0 if g_union == 0 else float(self.tot_inter / g_union)
        g_nsd = 1.0 if self.tot_surf_pixels == 0 else float(self.tot_surf_match / self.tot_surf_pixels)
        return {
            "global_dsc": g_dsc, "global_iou": g_iou, "global_nsd": g_nsd,
            "macro_dsc": float(np.mean(self.slice_dscs)), "macro_iou": float(np.mean(self.slice_ious)),
            "nsd": g_nsd,
        }


def evaluate_segmentation_batch(
    ground_truth: torch.Tensor, prediction: torch.Tensor, tau_threshold: float = 1.0, threshold: float = 0.5
) -> Dict[str, float]:
    """Compatibility wrapper returning batch mean scores."""
    acc = SegmentationMetricAccumulator(tau=tau_threshold, threshold=threshold)
    acc.update(ground_truth, prediction)
    res = acc.compute()
    return {"dsc": res["global_dsc"], "macro_dsc": res["macro_dsc"], "iou": res["global_iou"], "nsd": res["nsd"]}

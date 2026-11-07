"""Segmentation evaluation metrics: DSC, IoU, and NSD."""

from typing import Dict
import numpy as np
import scipy.ndimage as ndi
import torch


def compute_dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient (DSC) (Eq. 25)."""
    gt_bool = ground_truth.astype(bool)
    pred_bool = prediction.astype(bool)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    total_area = gt_bool.sum() + pred_bool.sum()
    if total_area == 0:
        return 1.0
    return float(2.0 * intersection / total_area)


def compute_intersection_over_union(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) (Eq. 26)."""
    gt_bool = ground_truth.astype(bool)
    pred_bool = prediction.astype(bool)
    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_normalized_surface_distance(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    tau_threshold: float = 1.0,
) -> float:
    """Compute Normalized Surface Distance (NSD) (Eq. 27).

    ASSUMPTION: Boundary extracted via morphological erosion difference;
    surface distance evaluated using Euclidean distance transform.
    """
    gt_bool = ground_truth.astype(bool)
    pred_bool = prediction.astype(bool)

    structure = ndi.generate_binary_structure(2, 1)
    boundary_gt = gt_bool ^ ndi.binary_erosion(gt_bool, structure=structure)
    boundary_pred = pred_bool ^ ndi.binary_erosion(pred_bool, structure=structure)

    num_gt_boundary = int(boundary_gt.sum())
    num_pred_boundary = int(boundary_pred.sum())

    if num_gt_boundary == 0 and num_pred_boundary == 0:
        return 1.0
    if num_gt_boundary == 0 or num_pred_boundary == 0:
        return 0.0

    dist_to_pred = ndi.distance_transform_edt(~boundary_pred)
    dist_to_gt = ndi.distance_transform_edt(~boundary_gt)

    gt_within_tau = np.logical_and(boundary_gt, dist_to_pred <= tau_threshold).sum()
    pred_within_tau = np.logical_and(boundary_pred, dist_to_gt <= tau_threshold).sum()

    return float((gt_within_tau + pred_within_tau) / (num_gt_boundary + num_pred_boundary))


def evaluate_segmentation_batch(
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    tau_threshold: float = 1.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate average DSC, IoU, and NSD for a batch."""
    gt_np = (ground_truth >= 0.5).detach().cpu().numpy().astype(np.uint8)
    pred_np = (prediction >= threshold).detach().cpu().numpy().astype(np.uint8)

    dsc_list, iou_list, nsd_list = [], [], []
    for i in range(gt_np.shape[0]):
        gt_slice = gt_np[i, 0]
        pred_slice = pred_np[i, 0]
        dsc_list.append(compute_dice_coefficient(gt_slice, pred_slice))
        iou_list.append(compute_intersection_over_union(gt_slice, pred_slice))
        nsd_list.append(compute_normalized_surface_distance(gt_slice, pred_slice, tau_threshold))

    return {
        "dsc": float(np.mean(dsc_list)),
        "iou": float(np.mean(iou_list)),
        "nsd": float(np.mean(nsd_list)),
    }


"""Shared utilities for pylidc multi-annotator alignment and exact fold-split indexing."""

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from mus_net.data.data_factory import _center_mask_area
from mus_net.data.patient_splitter import PatientWiseCrossValidationSplitter

NODULE_KEY_PATTERN = re.compile(r"^.+_scan(\d+)_nodule(\d+)$")


def ensure_pylidc_numpy_compat() -> None:
    """Restore deprecated NumPy aliases (np.int, np.float, np.bool) required by pylidc."""
    for attr, target in [("int", int), ("float", float), ("bool", bool)]:
        if not hasattr(np, attr):
            setattr(np, attr, target)


def parse_nodule_key(nodule_key: str) -> Tuple[int, int]:
    """Extract (scan_id, nodule_index) from a cache nodule_key string."""
    match = NODULE_KEY_PATTERN.match(str(nodule_key))
    if not match:
        raise ValueError(f"Unexpected nodule_key format: {nodule_key}")
    return int(match.group(1)), int(match.group(2))


def crop_centered_patch(image: np.ndarray, center_y: int, center_x: int, patch_size: int = 128) -> np.ndarray:
    """Pad 2D mask with zeros and crop square patch around (center_y, center_x)."""
    half = patch_size // 2
    padded = np.pad(image, half, mode="constant")
    return padded[center_y : center_y + patch_size, center_x : center_x + patch_size]


def extract_aligned_slice_mask(
    annotation: Any, absolute_z: int, center_y: int, center_x: int,
    patch_size: int = 128, canvas_size: int = 512,
) -> np.ndarray | None:
    """Reconstruct 2D binary mask of annotation at absolute_z aligned to ann0 crop center."""
    bbox = annotation.bbox()
    if absolute_z < bbox[2].start or absolute_z >= bbox[2].stop:
        return None
    mask_3d = annotation.boolean_mask()
    rel_z = absolute_z - bbox[2].start
    h = max(canvas_size, bbox[0].stop)
    w = max(canvas_size, bbox[1].stop)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[bbox[0], bbox[1]] = mask_3d[:, :, rel_z].astype(np.uint8)
    return crop_centered_patch(full_mask, center_y, center_x, patch_size)


def center_crop_64(mask_128: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Center-crop a 128x128 mask to target_size x target_size matching LidcIdriSegmentationDataset."""
    h, w = mask_128.shape[-2:]
    if h == target_size and w == target_size:
        return (mask_128 > 0).astype(np.uint8)
    sy, sx = (h - target_size) // 2, (w - target_size) // 2
    return (mask_128[..., sy : sy + target_size, sx : sx + target_size] > 0).astype(np.uint8)


def get_filtered_fold_val_indices(
    cache_path: Path, config: Dict[str, Any],
) -> List[List[int]]:
    """Return exact cache indices for each fold's validation split after dataset filtering."""
    dc = config["dataset"]
    min_ann = int(dc.get("min_annotators", 1))
    min_d, max_d = dc.get("min_diameter_mm"), dc.get("max_diameter_mm")
    p_size = int(dc.get("patch_size", 64))
    min_area = int(dc.get("min_mask_area") or 0)

    with np.load(cache_path) as c:
        masks, pids, anns, diams = c["masks"], c["patient_ids"], c["num_annotators"], c["diameter_mm"]

    selected_idx, selected_pids = [], []
    for i in range(len(masks)):
        a, d = int(anns[i]), float(diams[i])
        if a >= 0 and a < min_ann:
            continue
        if d >= 0.0 and ((min_d is not None and d < min_d) or (max_d is not None and d > max_d)):
            continue
        selected_idx.append(i)
        selected_pids.append(str(pids[i]))

    splitter = PatientWiseCrossValidationSplitter(dc["num_folds"], dc["seed"])
    folds = splitter.split(selected_pids)
    result = []
    for fold in folds:
        val_indices = [
            selected_idx[j] for j in fold["validation_indices"]
            if min_area <= 0 or _center_mask_area(masks[selected_idx[j]], p_size) >= min_area
        ]
        result.append(val_indices)
    return result

"""Extract nodule-centred 2D HU slices and first-annotation masks from a pylidc scan."""

from pathlib import Path
from typing import Any
import numpy as np


class NoduleSliceExtractor:
    """Crops every annotated slice of each nodule and stores filtering metadata."""

    def __init__(self, output_directory: Path, patch_size: int = 128) -> None:
        """Initialize output directory and half patch size."""
        self.output_directory = output_directory
        self.half_size = patch_size // 2

    def _crop_centered(self, image: np.ndarray, center_y: int, center_x: int, mode: str) -> np.ndarray:
        """Pad image by half patch size and crop a square patch around the center."""
        padded = np.pad(image, self.half_size, mode=mode)
        size = 2 * self.half_size
        return padded[center_y : center_y + size, center_x : center_x + size]

    def extract(self, scan: Any) -> int:
        """Save all slices of all nodules in scan; return number of saved slices."""
        volume = scan.to_volume()
        pixel_spacing = float(scan.pixel_spacing)
        saved_count = 0
        for nodule_index, annotations in enumerate(scan.cluster_annotations()):
            if not annotations:
                continue
            first_annotation = annotations[0]
            mask_3d, bounding_box = first_annotation.boolean_mask(), first_annotation.bbox()
            center_y = (bounding_box[0].start + bounding_box[0].stop) // 2
            center_x = (bounding_box[1].start + bounding_box[1].stop) // 2
            nodule_key = f"{scan.patient_id}_scan{scan.id}_nodule{nodule_index}"

            for relative_z in range(mask_3d.shape[2]):
                if not mask_3d[:, :, relative_z].any():
                    continue
                absolute_z = bounding_box[2].start + relative_z
                full_mask = np.zeros(volume.shape[:2], dtype=np.uint8)
                full_mask[bounding_box[0], bounding_box[1]] = mask_3d[:, :, relative_z]
                np.savez_compressed(
                    self.output_directory / f"{nodule_key}_slice{absolute_z}.npz",
                    hu_patch=self._crop_centered(volume[:, :, absolute_z], center_y, center_x, "edge").astype(np.int16),
                    mask=self._crop_centered(full_mask, center_y, center_x, "constant"),
                    patient_id=scan.patient_id,
                    nodule_key=nodule_key,
                    num_annotators=len(annotations),
                    diameter_mm=float(first_annotation.diameter),
                    pixel_spacing=pixel_spacing,
                )
                saved_count += 1
        return saved_count

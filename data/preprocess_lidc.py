"""Preprocess LIDC-IDRI scans into 2D HU patches using pylidc."""

from pathlib import Path
from typing import Optional
import numpy as np


def extract_lidc_patches(
    output_directory: str = "data/lidc_preprocessed",
    patch_size: int = 128,
    max_scans: Optional[int] = None,
) -> None:
    """Extract 2D slices around nodules using the 1st radiologist annotation.

    ASSUMPTION: Cropping 128x128 2D patches centered on nodule bounding boxes.
    Only the first annotation is used as ground truth as specified in Lin et al. (2025).
    """
    import pylidc as pl

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    if max_scans is not None:
        scans = scans[:max_scans]

    half_patch = patch_size // 2
    saved_counter = 0

    for scan in scans:
        patient_id = scan.patient_id
        nodules = scan.cluster_annotations()
        for nodule_index, annotations in enumerate(nodules):
            if not annotations:
                continue
            first_annotation = annotations[0]
            mask_3d = first_annotation.boolean_mask()
            bbox = first_annotation.bbox()

            vol = scan.to_volume()
            sub_vol = vol[bbox]

            for z_rel in range(mask_3d.shape[2]):
                slice_mask_2d = mask_3d[:, :, z_rel]
                if slice_mask_2d.sum() == 0:
                    continue

                z_abs = bbox[2].start + z_rel
                slice_hu = vol[:, :, z_abs]
                cy = int((bbox[0].start + bbox[0].stop) / 2)
                cx = int((bbox[1].start + bbox[1].stop) / 2)

                y1, y2 = max(0, cy - half_patch), min(slice_hu.shape[0], cy + half_patch)
                x1, x2 = max(0, cx - half_patch), min(slice_hu.shape[1], cx + half_patch)

                patch = np.pad(
                    slice_hu[y1:y2, x1:x2],
                    ((half_patch - (cy - y1), half_patch - (y2 - cy)),
                     (half_patch - (cx - x1), half_patch - (x2 - cx))),
                    mode="edge",
                )

                full_mask = np.zeros_like(slice_hu, dtype=np.uint8)
                full_mask[bbox[0], bbox[1]] = slice_mask_2d
                mask_patch = np.pad(
                    full_mask[y1:y2, x1:x2],
                    ((half_patch - (cy - y1), half_patch - (y2 - cy)),
                     (half_patch - (cx - x1), half_patch - (x2 - cx))),
                    mode="constant",
                )

                file_name = f"{patient_id}_nodule_{nodule_index}_slice_{z_abs}.npz"
                np.savez_compressed(
                    out_dir / file_name,
                    hu_patch=patch.astype(np.float32),
                    mask=mask_patch.astype(np.uint8),
                    patient_id=patient_id,
                )
                saved_counter += 1


if __name__ == "__main__":
    extract_lidc_patches()


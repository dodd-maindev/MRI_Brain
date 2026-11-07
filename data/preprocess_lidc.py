"""Preprocess LIDC-IDRI scans into 2D HU patches using pylidc."""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import mus_net.data.numpy_compatibility  # noqa: F401


def configure_pylidc(dicom_directory: str) -> None:
    """Create ~/.pylidcrc configuration file pointing to DICOM directory."""
    rc_path = Path.home() / ".pylidcrc"
    content = f"[dicom]\npath = {dicom_directory}\nwarn = True\n"
    rc_path.write_text(content, encoding="utf-8")


def extract_lidc_patches(
    output_dir: str = "data/lidc_preprocessed",
    patch_size: int = 128,
    max_scans: Optional[int] = None,
) -> None:
    """Extract 2D slices for 1st radiologist annotation across LIDC-IDRI."""
    import pylidc as pl

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    scans = pl.query(pl.Scan).all()
    if max_scans is not None:
        scans = scans[:max_scans]

    half = patch_size // 2
    for scan in scans:
        patient_id = scan.patient_id
        for nodule_idx, annotations in enumerate(scan.cluster_annotations()):
            if not annotations:
                continue
            first_ann = annotations[0]
            mask_3d, bbox = first_ann.boolean_mask(), first_ann.bbox()
            vol = scan.to_volume()

            for z_rel in range(mask_3d.shape[2]):
                slice_mask = mask_3d[:, :, z_rel]
                if slice_mask.sum() == 0:
                    continue
                z_abs = bbox[2].start + z_rel
                slice_hu = vol[:, :, z_abs]
                cy = int((bbox[0].start + bbox[0].stop) / 2)
                cx = int((bbox[1].start + bbox[1].stop) / 2)
                y1, y2 = max(0, cy - half), min(slice_hu.shape[0], cy + half)
                x1, x2 = max(0, cx - half), min(slice_hu.shape[1], cx + half)

                patch = np.pad(
                    slice_hu[y1:y2, x1:x2],
                    ((half - (cy - y1), half - (y2 - cy)), (half - (cx - x1), half - (x2 - cx))),
                    mode="edge",
                )
                full_mask = np.zeros_like(slice_hu, dtype=np.uint8)
                full_mask[bbox[0], bbox[1]] = slice_mask
                mask_patch = np.pad(
                    full_mask[y1:y2, x1:x2],
                    ((half - (cy - y1), half - (y2 - cy)), (half - (cx - x1), half - (x2 - cx))),
                    mode="constant",
                )
                np.savez_compressed(
                    out_path / f"{patient_id}_nodule_{nodule_idx}_slice_{z_abs}.npz",
                    hu_patch=patch.astype(np.int16),
                    mask=mask_patch.astype(np.uint8),
                    patient_id=patient_id,
                )


def main() -> None:
    """Parse CLI arguments for LIDC preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess LIDC-IDRI scans.")
    parser.add_argument("--dicom_dir", type=str, default=None, help="Root folder of DICOM files")
    parser.add_argument("--output_dir", type=str, default="data/lidc_preprocessed")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--max_scans", type=int, default=None)
    args = parser.parse_args()

    if args.dicom_dir:
        configure_pylidc(args.dicom_dir)
    extract_lidc_patches(args.output_dir, args.patch_size, args.max_scans)


if __name__ == "__main__":
    main()

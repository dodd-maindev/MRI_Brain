"""Streaming pipeline: download 1 patient -> crop int16 HU patch -> delete DICOM."""

from pathlib import Path
import shutil
from typing import List, Optional
import numpy as np
import mus_net.data.numpy_compatibility  # noqa: F401
from mus_net.data.tcia_downloader import TciaPatientScanDownloader


class StreamingLidcPreprocessor:
    """Processes LIDC-IDRI scans per-patient, keeping Colab disk usage under 500MB."""

    def __init__(self, output_dir: Path, temp_dir: Path, patch_size: int = 128) -> None:
        """Initialize paths, patch dimensions, and TCIA API client."""
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.patch_size = patch_size
        self.downloader = TciaPatientScanDownloader()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_patient_processed(self, patient_id: str) -> bool:
        """Check if patches for this patient already exist in the output directory."""
        return len(list(self.output_dir.glob(f"{patient_id}_nodule_*.npz"))) > 0

    def process_patient(self, patient_id: str) -> int:
        """Download single patient scan, crop 1st annotation 2D patches, delete DICOM."""
        if self.is_patient_processed(patient_id):
            return 0
        uids = self.downloader.get_ct_series_uids(patient_id)
        if not uids:
            return 0

        patient_temp = self.temp_dir / patient_id
        if not self.downloader.download_series_dicom(uids[0], patient_temp):
            shutil.rmtree(patient_temp, ignore_errors=True)
            return 0

        extracted_count = self._extract_patches_from_dicom(patient_temp, patient_id)
        shutil.rmtree(patient_temp, ignore_errors=True)
        return extracted_count

    def _extract_patches_from_dicom(self, dicom_dir: Path, patient_id: str) -> int:
        """Extract int16 HU nodule patches using 1st radiologist annotation via pylidc."""
        import pylidc as pl
        (Path.home() / ".pylidcrc").write_text(f"[dicom]\npath = {dicom_dir.parent}\nwarn = False\n")

        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if not scan:
            return 0

        count, half = 0, self.patch_size // 2
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
                    self.output_dir / f"{patient_id}_nodule_{nodule_idx}_slice_{z_abs}.npz",
                    hu_patch=patch.astype(np.int16),
                    mask=mask_patch.astype(np.uint8),
                    patient_id=patient_id,
                )
                count += 1
        return count

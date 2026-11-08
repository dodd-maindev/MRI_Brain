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
        """Initialize paths, patch dimensions, tracker file, and TCIA client."""
        self.output_dir, self.temp_dir = output_dir, temp_dir
        self.patch_size = patch_size
        self.downloader = TciaPatientScanDownloader()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file = self.output_dir / "processed_patients.txt"

    def is_patient_processed(self, patient_id: str) -> bool:
        """Check if patient has existing .npz files or is logged in tracker file."""
        if any(self.output_dir.glob(f"{patient_id}_nodule_*.npz")):
            return True
        if self.tracker_file.exists():
            return patient_id in set(self.tracker_file.read_text(encoding="utf-8").splitlines())
        return False

    def mark_patient_done(self, patient_id: str) -> None:
        """Record patient ID in persistent tracker file on Drive."""
        with open(self.tracker_file, "a", encoding="utf-8") as f:
            f.write(f"{patient_id}\n")

    def process_patient(self, patient_id: str) -> int:
        """Download scan, crop 1st annotation 2D patches, clean up temp DICOM safely."""
        if self.is_patient_processed(patient_id):
            return 0
        try:
            return self._download_and_extract(patient_id)
        except Exception as error:
            print(f"Warning: Patient {patient_id} encountered error: {error}. Skipping safely.")
            self.mark_patient_done(patient_id)
            return 0

    def _download_and_extract(self, patient_id: str) -> int:
        """Internal helper handling download, series matching, and patch extraction."""
        uids = self.downloader.get_ct_series_uids(patient_id)
        if not uids:
            self.mark_patient_done(patient_id)
            return 0

        patient_temp = self.temp_dir / patient_id
        if not self.downloader.download_series_dicom(uids[0], patient_temp):
            shutil.rmtree(patient_temp, ignore_errors=True)
            self.mark_patient_done(patient_id)
            return 0

        try:
            extracted_count = self._extract_patches_from_dicom(patient_temp, patient_id)
        finally:
            shutil.rmtree(patient_temp, ignore_errors=True)
        self.mark_patient_done(patient_id)
        return extracted_count

    def _extract_patches_from_dicom(self, dicom_dir: Path, patient_id: str) -> int:
        """Extract int16 HU patches using 1st annotation via pylidc (loads volume once)."""
        import pylidc as pl
        (Path.home() / ".pylidcrc").write_text(f"[dicom]\npath = {dicom_dir.parent}\nwarn = False\n")

        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
        if not scan:
            return 0

        vol = scan.to_volume()  # Optimization: Load 3D volume once per scan
        count, half = 0, self.patch_size // 2

        for nodule_idx, annotations in enumerate(scan.cluster_annotations()):
            if not annotations:
                continue
            first_ann = annotations[0]
            mask_3d, bbox = first_ann.boolean_mask(), first_ann.bbox()

            for z_rel in range(mask_3d.shape[2]):
                if mask_3d[:, :, z_rel].sum() == 0:
                    continue
                slice_hu = vol[:, :, bbox[2].start + z_rel]
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
                full_mask[bbox[0], bbox[1]] = mask_3d[:, :, z_rel]
                mask_patch = np.pad(
                    full_mask[y1:y2, x1:x2],
                    ((half - (cy - y1), half - (y2 - cy)), (half - (cx - x1), half - (x2 - cx))),
                    mode="constant",
                )
                np.savez_compressed(
                    self.output_dir / f"{patient_id}_nodule_{nodule_idx}_slice_{bbox[2].start + z_rel}.npz",
                    hu_patch=patch.astype(np.int16),
                    mask=mask_patch.astype(np.uint8),
                    patient_id=patient_id,
                )
                count += 1
        return count

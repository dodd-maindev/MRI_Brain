"""Streaming pipeline: download matching CT series -> crop int16 HU patches -> delete DICOM."""

from pathlib import Path
import shutil
import mus_net.data.numpy_compatibility  # noqa: F401
from mus_net.data.nodule_slice_extractor import NoduleSliceExtractor
from mus_net.data.tcia_downloader import TciaPatientScanDownloader


class StreamingLidcPreprocessor:
    """Processes LIDC-IDRI scans per patient, keeping Colab disk usage small."""

    def __init__(self, output_dir: Path, temp_dir: Path, patch_size: int = 128) -> None:
        """Initialize paths, tracker files, slice extractor, and TCIA client."""
        self.output_dir, self.temp_dir = output_dir, temp_dir
        self.downloader = TciaPatientScanDownloader()
        self.extractor = NoduleSliceExtractor(output_dir, patch_size)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file = self.output_dir / "processed_patients.txt"
        self.failure_file = self.output_dir / "failed_patients.txt"

    def is_patient_processed(self, patient_id: str) -> bool:
        """Check if patient has new-format .npz files or is logged as finished."""
        if any(self.output_dir.glob(f"{patient_id}_scan*.npz")):
            return True
        if self.tracker_file.exists():
            return patient_id in set(self.tracker_file.read_text(encoding="utf-8").splitlines())
        return False

    def _append_line(self, file_path: Path, text: str) -> None:
        """Append one line to a tracker file on Drive."""
        with open(file_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(f"{text}\n")

    def process_patient(self, patient_id: str) -> int:
        """Process all pylidc scans of a patient; failures are logged, not marked done."""
        if self.is_patient_processed(patient_id):
            return 0
        try:
            count = self._process_all_scans(patient_id)
        except Exception as error:  # noqa: BLE001
            print(f"Warning: {patient_id} failed: {error}. Logged for retry.")
            self._append_line(self.failure_file, f"{patient_id}\t{error}")
            return 0
        self._append_line(self.tracker_file, patient_id)
        return count

    def _process_all_scans(self, patient_id: str) -> int:
        """Download exactly the series referenced by pylidc annotations and extract slices."""
        import pylidc as pl

        (Path.home() / ".pylidcrc").write_text(f"[dicom]\npath = {self.temp_dir}\nwarn = False\n")
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        total = 0
        for scan in scans:
            if not scan.annotations:
                continue
            series_dir = self.temp_dir / patient_id / scan.series_instance_uid
            if not self.downloader.download_series_dicom(scan.series_instance_uid, series_dir):
                raise RuntimeError(f"download failed for series {scan.series_instance_uid}")
            try:
                total += self.extractor.extract(scan)
            finally:
                shutil.rmtree(self.temp_dir / patient_id, ignore_errors=True)
        return total

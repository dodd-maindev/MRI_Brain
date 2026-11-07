"""TCIA REST API client to download CT scans per patient."""

import io
from pathlib import Path
from typing import List, Optional
import zipfile
import requests


class TciaPatientScanDownloader:
    """Downloads DICOM CT series for a specific LIDC-IDRI patient via TCIA API."""

    BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

    def __init__(self, collection_name: str = "LIDC-IDRI") -> None:
        """Initialize collection identifier."""
        self.collection_name = collection_name

    def get_ct_series_uids(self, patient_id: str) -> List[str]:
        """Fetch SeriesInstanceUIDs for CT modality scans of given patient."""
        endpoint = f"{self.BASE_URL}/getSeries"
        params = {
            "Collection": self.collection_name,
            "PatientID": patient_id,
            "Modality": "CT",
        }
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        series_records = response.json()
        return [record["SeriesInstanceUID"] for record in series_records]

    def download_series_dicom(
        self, series_instance_uid: str, destination_dir: Path
    ) -> bool:
        """Download and extract zipped DICOM slice series to destination directory."""
        endpoint = f"{self.BASE_URL}/getImage"
        params = {"SeriesInstanceUID": series_instance_uid}
        destination_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(endpoint, params=params, timeout=120)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(destination_dir)
            return True
        except (requests.RequestException, zipfile.BadZipFile) as err:
            print(f"Failed to download series {series_instance_uid}: {err}")
            return False

    def fetch_all_patient_ids(self) -> List[str]:
        """Fetch list of all patient identifiers in the LIDC-IDRI collection."""
        endpoint = f"{self.BASE_URL}/getPatient"
        params = {"Collection": self.collection_name}
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        records = response.json()
        return sorted([record["PatientId"] for record in records])


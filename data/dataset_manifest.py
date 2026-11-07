"""Dataset manifest generator and integrity verifier for Google Drive."""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class DatasetManifestManager:
    """Manages manifest generation and integrity checks for preprocessed LIDC patches."""

    def __init__(self, dataset_directory: Path) -> None:
        """Initialize target directory containing preprocessed .npz files."""
        self.dataset_directory = dataset_directory

    def generate_manifest(self) -> Dict[str, Any]:
        """Inspect all .npz slices and create metadata JSON summary."""
        npz_files: List[Path] = sorted(list(self.dataset_directory.glob("*.npz")))
        if not npz_files:
            return {"total_slices": 0, "total_patients": 0, "status": "empty"}

        patient_to_slices: Dict[str, int] = {}
        for file_path in npz_files:
            patient_id = file_path.stem.split("_")[0]
            patient_to_slices[patient_id] = patient_to_slices.get(patient_id, 0) + 1

        first_sample = np.load(npz_files[0])
        manifest_data = {
            "total_slices": len(npz_files),
            "total_patients": len(patient_to_slices),
            "patch_shape": list(first_sample["hu_patch"].shape),
            "mask_shape": list(first_sample["mask"].shape),
            "hu_data_type": str(first_sample["hu_patch"].dtype),
            "mask_data_type": str(first_sample["mask"].dtype),
            "annotation_standard": "First radiologist annotation only (Lin et al. 2025)",
            "patient_distribution": patient_to_slices,
        }

        manifest_path = self.dataset_directory / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as file_handle:
            json.dump(manifest_data, file_handle, indent=2)

        return manifest_data

    def is_dataset_ready(self) -> bool:
        """Check if manifest exists and records valid slice entries."""
        manifest_path = self.dataset_directory / "dataset_manifest.json"
        if not manifest_path.exists():
            return False
        with open(manifest_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
        return data.get("total_slices", 0) > 0


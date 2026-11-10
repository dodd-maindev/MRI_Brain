"""Loads preprocessed LIDC .npz slices and applies nodule-level selection criteria."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class LidcSampleRepository:
    """Reads slices from disk and filters by annotator count and nodule diameter."""

    def __init__(
        self,
        data_directory: Path,
        min_annotators: int = 1,
        min_diameter_mm: Optional[float] = None,
        max_diameter_mm: Optional[float] = None,
    ) -> None:
        """Initialize directory and selection criteria (None disables a criterion)."""
        self.data_directory = data_directory
        self.min_annotators = min_annotators
        self.min_diameter_mm = min_diameter_mm
        self.max_diameter_mm = max_diameter_mm

    def has_data(self) -> bool:
        """Return True if the directory contains preprocessed slices."""
        return self.data_directory.exists() and any(self.data_directory.glob("*.npz"))

    def _is_selected(self, data: Any) -> bool:
        """Apply criteria; slices without metadata (old format) are always kept."""
        if "num_annotators" in data and int(data["num_annotators"]) < self.min_annotators:
            return False
        if "diameter_mm" in data:
            diameter = float(data["diameter_mm"])
            if self.min_diameter_mm is not None and diameter < self.min_diameter_mm:
                return False
            if self.max_diameter_mm is not None and diameter > self.max_diameter_mm:
                return False
        return True

    def load(self) -> List[Dict[str, Any]]:
        """Load and filter all slices, printing how many were kept."""
        files = sorted(self.data_directory.glob("*.npz"))
        samples: List[Dict[str, Any]] = []
        for file_path in files:
            data = np.load(file_path)
            if not self._is_selected(data):
                continue
            samples.append({
                "hu_patch": data["hu_patch"],
                "mask": data["mask"],
                "patient_id": str(data["patient_id"]) if "patient_id" in data else file_path.stem.split("_")[0],
            })
        print(f"Loaded {len(samples)}/{len(files)} slices after nodule selection criteria.")
        return samples

"""Loads preprocessed LIDC .npz slices with single-file caching and nodule filtering."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

CACHE_FILENAME = "lidc_packed_cache.npz"


class LidcSampleRepository:
    """Reads slices from cache or raw .npz files and filters by nodule metadata."""

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
        """Return True immediately if cache or at least one .npz slice exists."""
        if not self.data_directory.exists():
            return False
        if (self.data_directory / CACHE_FILENAME).exists():
            return True
        with os.scandir(self.data_directory) as entries:
            return any(e.name.endswith(".npz") and e.is_file() for e in entries)

    def _is_selected(self, num_annotators: int, diameter_mm: float) -> bool:
        """Apply nodule criteria; negative sentinel values mean metadata is absent."""
        if num_annotators >= 0 and num_annotators < self.min_annotators:
            return False
        if diameter_mm >= 0.0:
            if self.min_diameter_mm is not None and diameter_mm < self.min_diameter_mm:
                return False
            if self.max_diameter_mm is not None and diameter_mm > self.max_diameter_mm:
                return False
        return True

    def _load_from_cache(self, cache_path: Path) -> Tuple[List[Dict[str, Any]], int]:
        """Load pre-packed dataset cache in a single I/O operation."""
        print(f"Loading fast dataset cache from {cache_path.name}...", flush=True)
        with np.load(cache_path) as cache:
            hus, masks = cache["hu_patches"], cache["masks"]
            pids, anns, diams = cache["patient_ids"], cache["num_annotators"], cache["diameter_mm"]
        return [
            {"hu_patch": hus[i], "mask": masks[i], "patient_id": str(pids[i])}
            for i in range(len(hus)) if self._is_selected(int(anns[i]), float(diams[i]))
        ], len(hus)

    def _load_individual_and_cache(self) -> Tuple[List[Dict[str, Any]], int]:
        """Load per-slice .npz files with progress logs and write packed cache."""
        with os.scandir(self.data_directory) as entries:
            names = sorted(e.name for e in entries if e.name.endswith(".npz") and e.name != CACHE_FILENAME)
        hus, masks, pids, anns, diams = [], [], [], [], []
        for idx, name in enumerate(names, 1):
            if idx % 2000 == 0 or idx == len(names):
                print(f"  Reading slices from Drive: {idx}/{len(names)}...", flush=True)
            with np.load(self.data_directory / name) as data:
                hus.append(data["hu_patch"])
                masks.append(data["mask"])
                pids.append(str(data["patient_id"]) if "patient_id" in data else name.split("_")[0])
                anns.append(int(data["num_annotators"]) if "num_annotators" in data else -1)
                diams.append(float(data["diameter_mm"]) if "diameter_mm" in data else -1.0)

        if len(names) > 100:
            try:
                print(f"Saving single-file cache ({CACHE_FILENAME}) for fast future runs...", flush=True)
                np.savez_compressed(
                    self.data_directory / CACHE_FILENAME,
                    hu_patches=np.stack(hus), masks=np.stack(masks),
                    patient_ids=np.array(pids), num_annotators=np.array(anns, dtype=np.int16),
                    diameter_mm=np.array(diams, dtype=np.float32),
                )
            except OSError as err:
                print(f"Warning: could not write dataset cache ({err}).")

        return [
            {"hu_patch": hus[i], "mask": masks[i], "patient_id": pids[i]}
            for i in range(len(names)) if self._is_selected(anns[i], diams[i])
        ], len(names)

    def load(self) -> List[Dict[str, Any]]:
        """Load slices (from cache if present) and filter by selection criteria."""
        cache_path = self.data_directory / CACHE_FILENAME
        samples, total = (
            self._load_from_cache(cache_path) if cache_path.exists()
            else self._load_individual_and_cache()
        )
        print(f"Loaded {len(samples)}/{total} slices after nodule selection criteria.")
        return samples

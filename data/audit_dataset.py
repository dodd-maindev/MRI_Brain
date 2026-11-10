"""Audit a preprocessed LIDC directory: patients, nodules, slices, and mask statistics."""

from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import argparse
from collections import Counter
import numpy as np


class PreprocessedDatasetAuditor:
    """Reports dataset size and mask statistics to compare with the full LIDC-IDRI."""

    def __init__(self, data_directory: Path, crop_size: int = 64) -> None:
        """Initialize directory and network crop size used for truncation checks."""
        self.data_directory = data_directory
        self.crop_size = crop_size

    @staticmethod
    def _nodule_key_from_name(file_stem: str) -> str:
        """Strip the trailing slice index from a file stem to identify the nodule."""
        return file_stem.rsplit("_slice", 1)[0]

    def _count_tracker_lines(self, file_name: str) -> int:
        """Count non-empty lines in a tracker file if it exists."""
        path = self.data_directory / file_name
        return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]) if path.exists() else 0

    def run(self) -> None:
        """Print summary statistics."""
        files = sorted(self.data_directory.glob("*.npz"))
        patients = Counter(f.stem.split("_")[0] for f in files)
        nodules = Counter(self._nodule_key_from_name(f.stem) for f in files)
        areas, truncated = [], 0
        for file_path in files:
            mask = np.load(file_path)["mask"] > 0
            areas.append(int(mask.sum()))
            offset = (mask.shape[0] - self.crop_size) // 2
            inner = mask[offset : offset + self.crop_size, offset : offset + self.crop_size]
            truncated += int(inner.sum() < mask.sum())
        areas_array = np.array(areas) if areas else np.zeros(1)
        print("=== DATASET AUDIT ===")
        print(f"Slices (.npz)                 : {len(files)}")
        print(f"Patients with data            : {len(patients)}  (LIDC-IDRI has 1010 patients)")
        print(f"Nodules (first annotation)    : {len(nodules)}  (paper: 2635 annotated nodules)")
        print(f"Patients in processed tracker : {self._count_tracker_lines('processed_patients.txt')}")
        print(f"Patients in failure log       : {self._count_tracker_lines('failed_patients.txt')}")
        print(f"Slices per nodule (mean/max)  : {np.mean(list(nodules.values()) or [0]):.1f} / {max(nodules.values() or [0])}")
        print(f"Mask area px (p10/median/p90) : {np.percentile(areas_array, [10, 50, 90]).round(1).tolist()}")
        print(f"Masks < 20 px                 : {int((areas_array < 20).sum())}")
        print(f"Masks truncated by {self.crop_size}x{self.crop_size} crop : {truncated}")


def main() -> None:
    """Parse CLI arguments and run audit."""
    parser = argparse.ArgumentParser(description="Audit preprocessed LIDC patches.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    args = parser.parse_args()
    PreprocessedDatasetAuditor(Path(args.data_dir), args.crop_size).run()


if __name__ == "__main__":
    main()

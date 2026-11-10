"""Audit a preprocessed LIDC directory: patients, nodules, slices, and mask-HU alignment."""

from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import argparse
from collections import Counter
import numpy as np
import mus_net.data.numpy_compatibility  # noqa: F401


class PreprocessedDatasetAuditor:
    """Reports dataset size, mask statistics, centroid alignment, and HU ring contrast."""

    def __init__(self, data_directory: Path, crop_size: int = 64) -> None:
        """Initialize directory and network crop size used for truncation checks."""
        self.data_directory = data_directory
        self.crop_size = crop_size

    @staticmethod
    def _dilate_mask(mask: np.ndarray, radius: int = 4) -> np.ndarray:
        """Dilate binary 2D mask by radius pixels using pure NumPy shifts."""
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        height, width = mask.shape
        dilated = np.zeros_like(mask, dtype=bool)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    dilated |= padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
        return dilated

    def _count_tracker_lines(self, file_name: str) -> int:
        """Count non-empty lines in a tracker file if it exists."""
        path = self.data_directory / file_name
        return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]) if path.exists() else 0

    def run(self) -> None:
        """Inspect all .npz files and print comprehensive integrity and alignment report."""
        files = sorted(self.data_directory.glob("*.npz"))
        patients = Counter(f.stem.split("_")[0] for f in files)
        nodules = Counter(f.stem.rsplit("_slice", 1)[0] for f in files)
        areas, inside_hu, ring_hu, diffs, offsets, truncated = [], [], [], [], [], 0

        for file_path in files:
            sample = np.load(file_path)
            mask = sample["mask"] > 0
            hu = sample["hu_patch"].astype(np.float32)
            area = int(mask.sum())
            areas.append(area)
            pad = (mask.shape[0] - self.crop_size) // 2
            truncated += int(mask[pad : pad + self.crop_size, pad : pad + self.crop_size].sum() < area)
            if area >= 10:
                ring = self._dilate_mask(mask, radius=4) & (~mask)
                in_val, out_val = float(hu[mask].mean()), float(hu[ring].mean())
                inside_hu.append(in_val)
                ring_hu.append(out_val)
                diffs.append(in_val - out_val)
                ys, xs = np.nonzero(mask)
                center = mask.shape[0] / 2.0
                offsets.append(float(np.hypot(ys.mean() - center, xs.mean() - center)))

        areas_arr = np.array(areas) if areas else np.zeros(1)
        diffs_arr = np.array(diffs) if diffs else np.zeros(1)
        print(f"=== DATASET AUDIT: {self.data_directory.name} ===")
        print(f"Slices (.npz)                 : {len(files)}")
        print(f"Patients with data            : {len(patients)}  (LIDC-IDRI has 1010 patients)")
        print(f"Nodules (first annotation)    : {len(nodules)}  (paper: 2635 annotated nodules)")
        print(f"Patients in processed tracker : {self._count_tracker_lines('processed_patients.txt')}")
        print(f"Patients in failure log       : {self._count_tracker_lines('failed_patients.txt')}")
        print(f"Mask area px (p10/median/p90) : {np.percentile(areas_arr, [10, 50, 90]).round(1).tolist()}")
        print(f"Masks truncated by {self.crop_size}x{self.crop_size} crop : {truncated}")
        print(f"Mean HU inside mask           : {np.mean(inside_hu or [0]):.1f} HU")
        print(f"Mean HU in 4px surrounding    : {np.mean(ring_hu or [0]):.1f} HU")
        print(f"Contrast (Inside - Ring) med  : +{np.median(diffs_arr):.1f} HU (mean: +{np.mean(diffs_arr):.1f} HU)")
        print(f"Slices with Inside > Ring HU  : {(diffs_arr > 0).mean() * 100:.1f}%")
        print(f"Mean mask center offset (px)  : {np.mean(offsets or [0]):.2f} px from patch center")


def main() -> None:
    """Parse CLI arguments and run dataset audit."""
    parser = argparse.ArgumentParser(description="Audit preprocessed LIDC patches.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    args = parser.parse_args()
    PreprocessedDatasetAuditor(Path(args.data_dir), args.crop_size).run()


if __name__ == "__main__":
    main()

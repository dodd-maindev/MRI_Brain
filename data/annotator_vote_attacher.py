"""Attach multi-annotator masks and vote totals from packed cache to dataset samples."""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class AnnotatorVoteAttacher:
    """Loads multi-annotator mask cache once (mmap) and attaches per-slice votes."""

    def __init__(self, multi_cache_path: Path) -> None:
        """Initialize attacher with path to lidc_multi_annotator_cache.npz."""
        self.multi_cache_path = Path(multi_cache_path)

    def attach(self, samples: List[Dict[str, Any]]) -> None:
        """Attach annotator_masks (4, H, W) and annotator_count in-place by cache_index."""
        if not self.multi_cache_path.exists():
            raise FileNotFoundError(f"Multi-annotator cache not found: {self.multi_cache_path}")
        with np.load(self.multi_cache_path, mmap_mode="r") as cache:
            all_masks = np.asarray(cache["annotator_masks"])
        for sample in samples:
            idx = sample.get("cache_index")
            if idx is None:
                continue
            n_ann = min(max(1, int(sample.get("num_annotators", 1))), 4)
            sample["annotator_masks"] = all_masks[int(idx)]
            sample["annotator_count"] = n_ann

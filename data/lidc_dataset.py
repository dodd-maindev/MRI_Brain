"""LIDC-IDRI 2D CT dataset loader with multi-window generation."""

import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from mus_net.data.windowing import ComputedTomographyWindowTransformer


class LidcIdriSegmentationDataset(Dataset):
    """Dataset delivering multi-window CT slices and binary nodule masks."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        window_transformer: ComputedTomographyWindowTransformer,
        is_training: bool = False,
    ) -> None:
        """Initialize dataset with sample references and window transformer."""
        self.samples = samples
        self.transformer = window_transformer
        self.is_training = is_training

    def __len__(self) -> int:
        """Return total sample count."""
        return len(self.samples)

    def _apply_augmentation(
        self, hu_patch: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronous spatial augmentations to patch and mask."""
        if random.random() > 0.5:
            hu_patch = np.fliplr(hu_patch)
            mask = np.fliplr(mask)
        if random.random() > 0.5:
            hu_patch = np.flipud(hu_patch)
            mask = np.flipud(mask)
        k_rotations = random.choice([0, 1, 2, 3])
        if k_rotations > 0:
            hu_patch = np.rot90(hu_patch, k=k_rotations)
            mask = np.rot90(mask, k=k_rotations)
        return np.ascontiguousarray(hu_patch), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch item, transform windows, and format tensors."""
        item = self.samples[index]
        hu_patch = item["hu_patch"].astype(np.float32)
        mask = (item["mask"] > 0).astype(np.float32)

        if self.is_training:
            hu_patch, mask = self._apply_augmentation(hu_patch, mask)

        windows = self.transformer.transform(hu_patch)
        return {
            "raw": torch.from_numpy(windows["raw"]).unsqueeze(0).float(),
            "lung": torch.from_numpy(windows["lung"]).unsqueeze(0).float(),
            "medias": torch.from_numpy(windows["medias"]).unsqueeze(0).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "patient_id": str(item.get("patient_id", f"P_{index}")),
        }


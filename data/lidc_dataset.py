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
        target_patch_size: int = 64,
    ) -> None:
        """Initialize dataset with sample references, window transformer, and crop size."""
        self.samples = samples
        self.transformer = window_transformer
        self.is_training = is_training
        self.target_size = target_patch_size

    def __len__(self) -> int:
        """Return total sample count."""
        return len(self.samples)

    def _apply_augmentation(
        self, hu_patch: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronous spatial augmentations to patch and mask."""
        if random.random() > 0.5:
            hu_patch, mask = np.fliplr(hu_patch), np.fliplr(mask)
        if random.random() > 0.5:
            hu_patch, mask = np.flipud(hu_patch), np.flipud(mask)
        k_rot = random.choice([0, 1, 2, 3])
        if k_rot > 0:
            hu_patch, mask = np.rot90(hu_patch, k=k_rot), np.rot90(mask, k=k_rot)
        return np.ascontiguousarray(hu_patch), np.ascontiguousarray(mask)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch item, crop to target size, transform windows, and format tensors."""
        item = self.samples[index]
        hu_patch = item["hu_patch"].astype(np.float32)
        mask = (item["mask"] > 0).astype(np.float32)

        if self.target_size and hu_patch.shape[0] > self.target_size:
            h, w = hu_patch.shape
            sy, sx = (h - self.target_size) // 2, (w - self.target_size) // 2
            hu_patch = hu_patch[sy : sy + self.target_size, sx : sx + self.target_size]
            mask = mask[sy : sy + self.target_size, sx : sx + self.target_size]

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


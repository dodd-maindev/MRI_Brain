"""LIDC-IDRI 2D/2.5D Dual-FOV CT dataset loader with multi-window generation."""

import math, random
from typing import Any, Dict, List, Optional
import numpy as np, scipy.ndimage as ndi, torch
from torch.utils.data import Dataset
from mus_net.data.windowing import ComputedTomographyWindowTransformer


class LidcIdriSegmentationDataset(Dataset):
    """Dataset delivering multi-window (2D or 2.5D + Dual-FOV) CT slices and masks."""

    def __init__(
        self, samples: List[Dict[str, Any]], window_transformer: ComputedTomographyWindowTransformer,
        is_training: bool = False, target_patch_size: int = 64, use_25d: bool = False,
        use_dual_fov: bool = False, roi_crop_size: Optional[int] = None, random_annotator: bool = False,
    ) -> None:
        """Initialize dataset with sample list, window transformer, and ROI magnification."""
        self.samples, self.transformer = samples, window_transformer
        self.is_training, self.target_size = is_training, target_patch_size
        self.use_25d, self.use_dual_fov, self.roi_size = use_25d, use_dual_fov, roi_crop_size
        self.random_annotator = random_annotator

    def __len__(self) -> int:
        """Return total sample count."""
        return len(self.samples)

    def _crop_center(self, arr: np.ndarray) -> np.ndarray:
        """Center-crop to roi_size (or target_size) and resize to target_size."""
        crop_s = self.roi_size or self.target_size
        if crop_s and arr.shape[0] > crop_s:
            h, w = arr.shape
            sy, sx = (h - crop_s) // 2, (w - crop_s) // 2
            arr = arr[sy : sy + crop_s, sx : sx + crop_s]
        if self.target_size and arr.shape[0] != self.target_size:
            arr = ndi.zoom(arr, self.target_size / float(arr.shape[0]), order=1).astype(np.float32)
        return arr

    def _downsample_global(self, hu_full: np.ndarray) -> np.ndarray:
        """Resize full 128x128 FOV to target_size for wide anatomical context."""
        if not self.target_size or hu_full.shape[0] == self.target_size:
            return hu_full
        zoom_f = self.target_size / float(hu_full.shape[0])
        return ndi.zoom(hu_full, zoom_f, order=1).astype(np.float32)

    def _apply_augmentation(
        self, hu_stack: np.ndarray, mask_stack: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply synchronous spatial augmentations to [C, H, W] stack and [M, H, W] masks."""
        if random.random() > 0.5: hu_stack, mask_stack = np.flip(hu_stack, axis=2), np.flip(mask_stack, axis=2)
        if random.random() > 0.5: hu_stack, mask_stack = np.flip(hu_stack, axis=1), np.flip(mask_stack, axis=1)
        k_rot = random.choice([0, 1, 2, 3])
        if k_rot > 0: hu_stack, mask_stack = np.rot90(hu_stack, k=k_rot, axes=(1, 2)), np.rot90(mask_stack, k=k_rot, axes=(1, 2))
        if random.random() > 0.65:
            ang, sc = math.radians(random.uniform(-10.0, 10.0)), random.uniform(0.90, 1.10)
            ca, sa = math.cos(ang) / sc, math.sin(ang) / sc
            mat = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            ctr = np.array(mask_stack.shape[1:], dtype=np.float32) / 2.0 - 0.5
            off = ctr - mat @ ctr
            hu_stack = np.stack([ndi.affine_transform(ch, mat, offset=off, order=1, mode="nearest") for ch in hu_stack])
            mask_stack = np.stack([(ndi.affine_transform(m, mat, offset=off, order=0, mode="constant") > 0.5).astype(np.float32) for m in mask_stack])
        if random.random() > 0.5:
            ms, (_, h, w) = 4, mask_stack.shape
            sy, sx = ms + random.randint(-ms, ms), ms + random.randint(-ms, ms)
            hu_stack = np.pad(hu_stack, ((0, 0), (ms, ms), (ms, ms)), mode="edge")[:, sy : sy + h, sx : sx + w]
            mask_stack = np.pad(mask_stack, ((0, 0), (ms, ms), (ms, ms)), mode="constant")[:, sy : sy + h, sx : sx + w]
        return np.ascontiguousarray(hu_stack), np.ascontiguousarray(mask_stack)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Build [C, H, W] multi-window tensors, mask, and optional multi-annotator votes."""
        item = self.samples[index]
        hu_full = item["hu_patch"].astype(np.float32)
        mask = (self._crop_center((item["mask"] > 0).astype(np.float32)) >= 0.5).astype(np.float32)
        has_multi = "annotator_masks" in item
        mask_stack = np.stack([mask] + [(self._crop_center((m > 0).astype(np.float32)) >= 0.5).astype(np.float32) for m in item["annotator_masks"]], axis=0) if has_multi else mask[None]
        channels = (
            [self._crop_center(item.get("prev_hu", hu_full).astype(np.float32)), self._crop_center(hu_full), self._crop_center(item.get("next_hu", hu_full).astype(np.float32))]
            if self.use_25d else [self._crop_center(hu_full)]
        )
        if self.use_dual_fov:
            channels.append(self._downsample_global(hu_full))
        hu_stack = np.stack(channels, axis=0)
        if self.is_training:
            hu_stack, mask_stack = self._apply_augmentation(hu_stack, mask_stack)
        windows = self.transformer.transform(hu_stack)
        out = {
            "raw": torch.from_numpy(windows["raw"]).float(), "lung": torch.from_numpy(windows["lung"]).float(),
            "medias": torch.from_numpy(windows["medias"]).float(), "mask": torch.from_numpy(mask_stack[0]).unsqueeze(0).float(),
            "patient_id": str(item.get("patient_id", f"P_{index}")),
        }
        if has_multi:
            n_ann = int(item.get("annotator_count", 1))
            ann_s = mask_stack[1:5]
            out["vote_positive"] = torch.from_numpy(ann_s[:n_ann].sum(axis=0, keepdims=True)).float()
            out["vote_total"] = torch.tensor([[[float(n_ann)]]], dtype=torch.float32)
            if self.random_annotator and self.is_training:
                k = random.randrange(n_ann)
                out["mask_random"] = torch.from_numpy(ann_s[k : k + 1]).float()
        return out

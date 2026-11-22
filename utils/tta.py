"""Test-Time Augmentation (D4 8x) and Largest Connected Component post-processing."""

from typing import Dict
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn


def predict_with_d4_tta(
    model: nn.Module, inputs: Dict[str, torch.Tensor], enabled: bool = True
) -> torch.Tensor:
    """Average model final_prediction across 8 flip/rotation symmetries."""
    if not enabled:
        return model(inputs)["final_prediction"]
    acc = torch.zeros_like(inputs["raw"][:, :1])
    for k in range(4):
        for flip in (False, True):
            aug = {name: torch.rot90(tensor, k, (-2, -1)) for name, tensor in inputs.items()}
            if flip:
                aug = {name: torch.flip(tensor, (-1,)) for name, tensor in aug.items()}
            pred = model(aug)["final_prediction"]
            if flip:
                pred = torch.flip(pred, (-1,))
            acc = acc + torch.rot90(pred, -k, (-2, -1))
    return acc / 8.0


def suppress_stray_components(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Retain the main nodule component per slice to eliminate stray FP specks."""
    p_np = probs.detach().cpu().numpy().copy()
    for i in range(p_np.shape[0]):
        labeled, num = ndi.label(p_np[i, 0] >= threshold)
        if num > 1:
            counts = np.bincount(labeled.ravel())
            counts[0] = 0
            largest = int(np.argmax(counts))
            p_np[i, 0][labeled != largest] = 0.0
    return torch.from_numpy(p_np).to(probs.device)

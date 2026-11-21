"""Test-Time Augmentation (D4 dihedral 8x symmetry averaging) for inference."""

from typing import Dict
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

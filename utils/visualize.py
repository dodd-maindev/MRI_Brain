"""Visualization utilities for multi-window CT masks and MWM maps."""

from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_segmentation_comparison(
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    output_path: Path,
    gradcam_heatmap: Optional[torch.Tensor] = None,
) -> None:
    """Save comprehensive multi-panel figure illustrating all intermediate predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_columns = 8 if gradcam_heatmap is not None else 7
    fig, axes = plt.subplots(1, num_columns, figsize=(3 * num_columns, 3.5))

    img_raw = inputs["raw"][0, 0].detach().cpu().numpy()
    img_lung = inputs["lung"][0, 0].detach().cpu().numpy()
    img_medias = inputs["medias"][0, 0].detach().cpu().numpy()
    gt = ground_truth[0, 0].detach().cpu().numpy()
    final_pred = outputs["final_prediction"][0, 0].detach().cpu().numpy()

    axes[0].imshow(img_raw, cmap="gray")
    axes[0].set_title("Raw Input")
    axes[1].imshow(img_lung, cmap="gray")
    axes[1].set_title("Lung Window")
    axes[2].imshow(img_medias, cmap="gray")
    axes[2].set_title("Medias Window")

    # MWM rendering: RGB array [Red=medias-only, Green=lung-only, Blue=common]
    if outputs.get("multi_window_mask") is not None:
        mwm_tensor = outputs["multi_window_mask"][0].detach().cpu().numpy()
        mwm_rgb = np.transpose(mwm_tensor, (1, 2, 0))
        axes[3].imshow(np.clip(mwm_rgb, 0.0, 1.0))
        axes[3].set_title("MWM (G:Lung, R:Med)")
    else:
        axes[3].text(0.5, 0.5, "No MWM", ha="center")
        axes[3].set_title("MWM")

    lung_pred = outputs["branch_masks"].get("lung")
    if lung_pred is not None:
        axes[4].imshow(lung_pred[0, 0].detach().cpu().numpy(), cmap="hot")
    axes[4].set_title("S_lung")

    axes[5].imshow(final_pred >= 0.5, cmap="gray")
    axes[5].set_title("Final Pred")
    axes[6].imshow(gt, cmap="gray")
    axes[6].set_title("Ground Truth")

    if gradcam_heatmap is not None:
        cam_np = gradcam_heatmap[0, 0].detach().cpu().numpy()
        axes[7].imshow(img_raw, cmap="gray")
        axes[7].imshow(cam_np, cmap="jet", alpha=0.5)
        axes[7].set_title("Grad-CAM")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


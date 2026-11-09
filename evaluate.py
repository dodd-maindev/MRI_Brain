"""Evaluate trained MUS-Net models, generate tables and Grad-CAM visualizations."""

from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import argparse
import numpy as np
import torch
import yaml
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.train import build_network_from_config
from mus_net.utils.gradcam import GradCamExplainer
from mus_net.utils.visualize import save_segmentation_comparison


def evaluate_single_fold(model: torch.nn.Module, val_loader: any, tau: float, threshold: float = 0.5) -> dict:
    """Compute Global Volumetric (Eq. 25-27) and Macro scores on a validation fold."""
    device = next(model.parameters()).device
    model.eval()
    accumulator = SegmentationMetricAccumulator(tau=tau, threshold=threshold)

    for batch in val_loader:
        inputs = {k: batch[k].to(device) for k in ["raw", "lung", "medias"]}
        masks = batch["mask"].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        accumulator.update(masks, outputs["final_prediction"])

    return accumulator.compute()


def export_sample_visualizations(model: torch.nn.Module, val_loader: any, output_dir: Path) -> None:
    """Export MWM and Grad-CAM figure for the first batch."""
    device = next(model.parameters()).device
    batch = next(iter(val_loader))
    inputs = {k: batch[k].to(device) for k in ["raw", "lung", "medias"]}
    masks = batch["mask"].to(device)

    target_layer = model.fusion_head[2].block[3]
    explainer = GradCamExplainer(model, target_layer)
    outputs = model(inputs)
    cam = explainer.generate_cam(outputs["final_prediction"])
    save_segmentation_comparison(inputs, outputs, masks, output_dir / "sample_eval.png", cam)


def main() -> None:
    """Parse CLI arguments and report Volumetric & Macro metrics matching Table 1."""
    parser = argparse.ArgumentParser(description="Evaluate MUS-Net checkpoints.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--drive_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config["dataset"]["data_directory"] = args.data_dir
    if args.drive_dir:
        d_p = Path(args.drive_dir)
        config["train"]["checkpoint_directory"] = str(d_p / "checkpoints")
        config["evaluation"]["visualization_directory"] = str(d_p / "visualizations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = config["evaluation"]["normalized_surface_distance_tau"]
    vis_dir = Path(config["evaluation"]["visualization_directory"])
    ckpt_dir = Path(config["train"]["checkpoint_directory"])

    print(f"\nEvaluating with Threshold = {args.threshold:.2f} (Paper Table 1 standard)")
    print("| Fold | Global DSC (%) | IoU (%) | NSD (%) | Macro DSC (%) |\n|------|----------------|---------|---------|---------------|")

    fold_metrics = []
    for f in range(config["dataset"]["num_folds"]):
        ckpt_path = ckpt_dir / f"best_model_fold_{f}.pth"
        if not ckpt_path.exists():
            continue
        model = build_network_from_config(config).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
        _, val_loader = prepare_fold_dataloaders(config, f)
        res = evaluate_single_fold(model, val_loader, tau, args.threshold)
        fold_metrics.append(res)
        print(f"| {f:4d} | {res['global_dsc']*100:14.2f} | {res['global_iou']*100:7.2f} | {res['nsd']*100:7.2f} | {res['macro_dsc']*100:13.2f} |")
        if f == 0 and config["evaluation"]["export_visualizations"]:
            export_sample_visualizations(model, val_loader, vis_dir)

    if fold_metrics:
        m_d, s_d = np.mean([m["global_dsc"] for m in fold_metrics]) * 100, np.std([m["global_dsc"] for m in fold_metrics]) * 100
        m_i, m_n = np.mean([m["global_iou"] for m in fold_metrics]) * 100, np.mean([m["nsd"] for m in fold_metrics]) * 100
        m_macro = np.mean([m["macro_dsc"] for m in fold_metrics]) * 100
        print(f"| Mean | {m_d:.2f} +/- {s_d:.2f} | {m_i:.2f} | {m_n:.2f} | {m_macro:.2f} |\n")


if __name__ == "__main__":
    main()

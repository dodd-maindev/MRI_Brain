"""Evaluate trained MUS-Net models, generate tables and Grad-CAM visualizations."""

import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import evaluate_segmentation_batch
from mus_net.train import build_network_from_config
from mus_net.utils.gradcam import GradCamExplainer
from mus_net.utils.visualize import save_segmentation_comparison


def evaluate_single_fold(model: torch.nn.Module, val_loader: any, tau: float) -> dict:
    """Compute mean DSC, IoU, and NSD on a validation fold."""
    device = next(model.parameters()).device
    model.eval()
    dsc_list, iou_list, nsd_list = [], [], []

    for batch in val_loader:
        inputs = {k: batch[k].to(device) for k in ["raw", "lung", "medias"]}
        masks = batch["mask"].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        metrics = evaluate_segmentation_batch(masks, outputs["final_prediction"], tau)
        dsc_list.append(metrics["dsc"])
        iou_list.append(metrics["iou"])
        nsd_list.append(metrics["nsd"])

    return {"dsc": float(np.mean(dsc_list)), "iou": float(np.mean(iou_list)), "nsd": float(np.mean(nsd_list))}


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
    """Parse arguments and execute multi-fold evaluation with optional Drive support."""
    parser = argparse.ArgumentParser(description="Evaluate MUS-Net checkpoints.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed .npz data")
    parser.add_argument("--drive_dir", type=str, default=None, help="Root path in Google Drive")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        config["dataset"]["data_directory"] = args.data_dir
    if args.drive_dir:
        d_path = Path(args.drive_dir)
        config["train"]["checkpoint_directory"] = str(d_path / "checkpoints")
        config["evaluation"]["visualization_directory"] = str(d_path / "visualizations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = config["evaluation"]["normalized_surface_distance_tau"]
    vis_dir = Path(config["evaluation"]["visualization_directory"])
    ckpt_dir = Path(config["train"]["checkpoint_directory"])

    fold_metrics = []
    print("\n| Fold | DSC (%) | IoU (%) | NSD (%) |")
    print("|------|---------|---------|---------|")

    for f in range(config["dataset"]["num_folds"]):
        model = build_network_from_config(config).to(device)
        ckpt_path = ckpt_dir / f"best_model_fold_{f}.pth"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        _, val_loader = prepare_fold_dataloaders(config, f)
        res = evaluate_single_fold(model, val_loader, tau)
        fold_metrics.append(res)
        print(f"| {f:4d} | {res['dsc']*100:7.2f} | {res['iou']*100:7.2f} | {res['nsd']*100:7.2f} |")
        if f == 0 and config["evaluation"]["export_visualizations"]:
            export_sample_visualizations(model, val_loader, vis_dir)

    mean_dsc, mean_iou = np.mean([m["dsc"] for m in fold_metrics]) * 100, np.mean([m["iou"] for m in fold_metrics]) * 100
    mean_nsd, std_dsc = np.mean([m["nsd"] for m in fold_metrics]) * 100, np.std([m["dsc"] for m in fold_metrics]) * 100
    print(f"| Mean | {mean_dsc:.2f}+-{std_dsc:.2f} | {mean_iou:.2f} | {mean_nsd:.2f} |\n")


if __name__ == "__main__":
    main()

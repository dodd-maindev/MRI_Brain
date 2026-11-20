"""Evaluate trained MUS-Net models, generate tables and Grad-CAM visualizations."""

from pathlib import Path
import argparse, sys, numpy as np, torch, yaml

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path: sys.path.insert(0, _parent)
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.train import build_network_from_config
from mus_net.utils.gradcam import GradCamExplainer
from mus_net.utils.q1_metrics import compute_calibration_and_aurc, compute_slice_hd95
from mus_net.utils.visualize import save_segmentation_comparison


def evaluate_single_fold(model: torch.nn.Module, val_loader: any, tau: float, threshold: float = 0.5) -> dict:
    """Compute Global Volumetric, Macro, HD95, ECE, and AURC on a validation fold."""
    device = next(model.parameters()).device
    model.eval()
    accumulator = SegmentationMetricAccumulator(tau=tau, threshold=threshold)
    probs_list, gts_list, hd95s = [], [], []
    for batch in val_loader:
        inputs = {k: batch[k].to(device) for k in ["raw", "lung", "medias"]}
        masks = batch["mask"].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        pred = outputs["final_prediction"]
        accumulator.update(masks, pred)
        p_np, g_np = pred.cpu().numpy(), masks.cpu().numpy()
        probs_list.append(p_np); gts_list.append(g_np)
        for i in range(p_np.shape[0]):
            hd95s.append(compute_slice_hd95(g_np[i, 0] >= 0.5, p_np[i, 0] >= threshold))
    res = accumulator.compute()
    cal = compute_calibration_and_aurc(np.concatenate(probs_list), np.concatenate(gts_list))
    res.update({"hd95": float(np.mean(hd95s)), "ece": cal["ece"], "aurc": cal["aurc"]})
    return res


def export_sample_visualizations(model: torch.nn.Module, val_loader: any, output_dir: Path) -> None:
    """Export MWM and Grad-CAM figure for the first batch."""
    device = next(model.parameters()).device
    batch = next(iter(val_loader))
    inputs = {k: batch[k].to(device) for k in ["raw", "lung", "medias"]}
    masks = batch["mask"].to(device)

    target_layer = model.fusion_head[2].block[3]
    explainer = GradCamExplainer(model, target_layer)
    outputs = model(inputs)
    cam = explainer.generate_cam(outputs["final_prediction"], spatial_size=tuple(masks.shape[-2:]))
    save_segmentation_comparison(inputs, outputs, masks, output_dir / "sample_eval.png", cam)


def main() -> None:
    """Parse CLI arguments and report Volumetric & Macro metrics matching Table 1."""
    parser = argparse.ArgumentParser(description="Evaluate MUS-Net checkpoints.")
    for arg, kw in [("--config", {"default": "configs/default.yaml"}), ("--data_dir", {"default": None}),
                    ("--drive_dir", {"default": None}), ("--threshold", {"type": float, "default": 0.5})]:
        parser.add_argument(arg, **kw)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.data_dir: config["dataset"]["data_directory"] = args.data_dir
    if args.drive_dir:
        d_p = Path(args.drive_dir)
        config["train"]["checkpoint_directory"] = str(d_p / "checkpoints")
        config["evaluation"]["visualization_directory"] = str(d_p / "visualizations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau = config["evaluation"]["normalized_surface_distance_tau"]
    vis_dir = Path(config["evaluation"]["visualization_directory"])
    ckpt_dir = Path(config["train"]["checkpoint_directory"])

    print(f"\nEvaluating with Threshold = {args.threshold:.2f} (Paper Table 1 + Q1 Metrics)")
    print("| Fold | DSC (%) | IoU (%) | NSD (%) | HD95 | ECE | AURC | Macro DSC |\n|------|---------|---------|---------|------|-----|------|-----------|")

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
        print(f"| {f:4d} | {res['global_dsc']*100:7.2f} | {res['global_iou']*100:7.2f} | {res['nsd']*100:7.2f} | {res['hd95']:4.2f} | {res['ece']:.4f} | {res['aurc']:.4f} | {res['macro_dsc']*100:9.2f} |")
        if f == 0 and config["evaluation"]["export_visualizations"]:
            export_sample_visualizations(model, val_loader, vis_dir)

    if fold_metrics:
        m_d, s_d = np.mean([m["global_dsc"] for m in fold_metrics]) * 100, np.std([m["global_dsc"] for m in fold_metrics]) * 100
        m_i, m_n = np.mean([m["global_iou"] for m in fold_metrics]) * 100, np.mean([m["nsd"] for m in fold_metrics]) * 100
        m_h, m_e, m_a = np.mean([m["hd95"] for m in fold_metrics]), np.mean([m["ece"] for m in fold_metrics]), np.mean([m["aurc"] for m in fold_metrics])
        print(f"| Mean | {m_d:.2f}+/-{s_d:.2f} | {m_i:.2f} | {m_n:.2f} | {m_h:.2f} | {m_e:.4f} | {m_a:.4f} | {np.mean([m['macro_dsc'] for m in fold_metrics])*100:.2f} |\n")


if __name__ == "__main__":
    main()

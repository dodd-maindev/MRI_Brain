"""Entry point for training MUS-Net with real-time Drive saving and resume."""

from pathlib import Path
import argparse, sys, numpy as np, torch, yaml

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path: sys.path.insert(0, _parent)
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.models.mus_net import MultiWindowSegmentationNetwork
from mus_net.trainer import MusNetTrainer


def build_network_from_config(config: dict) -> MultiWindowSegmentationNetwork:
    """Instantiate MUS-Net or AMU-Net using parameters from YAML config."""
    dc, m, ab = config["dataset"], config["model"], config["ablation"]
    in_ch = (3 if dc.get("use_25d", False) else 1) + (1 if dc.get("use_dual_fov", False) else 0)
    return MultiWindowSegmentationNetwork(
        base_channels=m["base_channels"], latent_dimension=m["latent_dimension"], beta=m["beta"],
        attention_downsample_ratio=m["attention_downsample_ratio"],
        lung_filter=ab["lung_filter"], medias_filter=ab["medias_filter"], raw_filter=ab["raw_filter"],
        active_branches=ab["active_branches"],
        use_uim=ab["use_uncertainty_indication_module"], use_ffm=ab["use_feature_filter_module"],
        use_uncertainty_gate=ab.get("use_uncertainty_gate", False), use_adaptive_window=ab.get("use_adaptive_window", False),
        otsu_soft=m.get("otsu_soft", False), otsu_temperature=m.get("otsu_temperature", 8.0),
        unet_stages=m.get("unet_stages", 5), use_aspp_bottleneck=m.get("use_aspp_bottleneck", False),
        dropout_rate=m.get("dropout_rate", 0.0), in_channels=in_ch,
        use_evidential=ab.get("use_evidential", False),
    )


def execute_fold_training(config: dict, fold: int, reset: bool = False) -> float:
    """Run training lifecycle with automatic resume and immediate Drive syncing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_loader, val_loader = prepare_fold_dataloaders(config, fold)
    model = build_network_from_config(config)
    trainer = MusNetTrainer(model, config, fold)

    start_epoch = 1 if reset else trainer.resume_if_available()
    num_epochs = config["train"]["epochs"]
    if start_epoch > num_epochs:
        return trainer.best_dsc

    for epoch in range(start_epoch, num_epochs + 1):
        tr_loss = trainer.train_epoch(train_loader, epoch)
        v = trainer.validate(val_loader)
        trainer.log_epoch(epoch, tr_loss, v["loss"], v["dsc"], v["iou"], v["nsd"])
        trainer.save_checkpoint(v["dsc"], epoch)
        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | Train: {tr_loss:.4f} | "
            f"Val: {v['loss']:.4f} | DSC: {v['dsc']*100:.2f}% | "
            f"IoU: {v['iou']*100:.2f}% | NSD: {v['nsd']*100:.2f}% | Best: {trainer.best_dsc*100:.2f}%",
            flush=True,
        )
    return trainer.best_dsc


def main() -> None:
    """Parse CLI arguments, redirect Drive directories, and run training."""
    parser = argparse.ArgumentParser(description="Train MUS-Net on LIDC-IDRI.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None, help="Fold index (0-4)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed .npz data")
    parser.add_argument("--drive_dir", type=str, default=None, help="Root path in Google Drive")
    parser.add_argument("--reset", action="store_true", help="Start training fresh from epoch 1")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        config["dataset"]["data_directory"] = args.data_dir
    if args.drive_dir:
        d_path = Path(args.drive_dir)
        config["train"]["checkpoint_directory"] = str(d_path / "checkpoints")
        config["train"]["log_directory"] = str(d_path / "logs")
        config["evaluation"]["visualization_directory"] = str(d_path / "visualizations")

    num_folds = config["dataset"]["num_folds"]
    target_folds = [args.fold] if args.fold is not None else list(range(num_folds))

    scores = [execute_fold_training(config, f, args.reset) for f in target_folds]
    print(f"\nFinal: Mean DSC = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")


if __name__ == "__main__":
    main()

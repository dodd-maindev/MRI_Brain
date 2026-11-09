"""Entry point for training MUS-Net with real-time Drive saving and resume."""

from pathlib import Path
import sys

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import argparse
import numpy as np
import yaml
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.models.mus_net import MultiWindowSegmentationNetwork
from mus_net.trainer import MusNetTrainer


def build_network_from_config(config: dict) -> MultiWindowSegmentationNetwork:
    """Instantiate MUS-Net using parameters from YAML config."""
    return MultiWindowSegmentationNetwork(
        base_channels=config["model"]["base_channels"],
        latent_dimension=config["model"]["latent_dimension"],
        beta=config["model"]["beta"],
        attention_downsample_ratio=config["model"]["attention_downsample_ratio"],
        lung_filter=config["ablation"]["lung_filter"],
        medias_filter=config["ablation"]["medias_filter"],
        raw_filter=config["ablation"]["raw_filter"],
        active_branches=config["ablation"]["active_branches"],
        use_uim=config["ablation"]["use_uncertainty_indication_module"],
        use_ffm=config["ablation"]["use_feature_filter_module"],
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
        tr_loss = trainer.train_epoch(train_loader)
        v = trainer.validate(val_loader)
        trainer.log_epoch(epoch, tr_loss, v["loss"], v["dsc"], v["iou"], v["nsd"])
        trainer.save_checkpoint(v["dsc"], epoch)

        if epoch % 5 == 0 or epoch == start_epoch:
            print(
                f"Epoch {epoch:03d} | Train: {tr_loss:.4f} | "
                f"Val: {v['loss']:.4f} | DSC: {v['dsc']:.4f} | "
                f"IoU: {v['iou']:.4f} | Best: {trainer.best_dsc:.4f}"
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

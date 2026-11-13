"""Trainer class managing optimization, immediate Drive checkpointing, and resume."""

import csv
from pathlib import Path
from typing import Any, Dict
import torch
import torch.optim as optim
from mus_net.losses import MultiWindowSegmentationLoss
from mus_net.metrics import SegmentationMetricAccumulator


class MusNetTrainer:
    """Encapsulates training, validation, real-time logging, and resume logic."""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], fold_idx: int) -> None:
        """Initialize optimizer, loss criterion, checkpoints, and log file."""
        self.config, self.fold_idx = config, fold_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        lr, opt = config["train"]["learning_rate"], config["train"].get("optimizer", "adam").lower()
        self.optimizer = (optim.SGD(self.model.parameters(), lr=lr, momentum=config["train"].get("momentum", 0.9))
                          if opt == "sgd" else optim.Adam(self.model.parameters(), lr=lr))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config["train"]["epochs"], eta_min=1e-6)
        lc = config["loss"]
        self.criterion = MultiWindowSegmentationLoss(
            lc["alpha_1"], lc["alpha_2"], lc.get("use_dice_loss", True), lc.get("dice_weight", 1.0),
            lc.get("bce_weight", 1.0), lc["use_posterior_kl"], lc["kl_weight"]
        )
        self.best_dsc, self.ckpt_dir = -1.0, Path(config["train"]["checkpoint_directory"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(config["train"]["log_directory"]) / f"log_fold_{fold_idx}.csv"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dsc", "val_iou", "val_nsd"])

    def log_epoch(self, epoch: int, tr_loss: float, v_loss: float, dsc: float, iou: float, nsd: float) -> None:
        """Append metrics to CSV and flush to Drive immediately."""
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{v_loss:.4f}", f"{dsc:.4f}", f"{iou:.4f}", f"{nsd:.4f}"])
            f.flush()

    def resume_if_available(self) -> int:
        """Restore model and training state, prioritizing higher DSC checkpoint."""
        b_p, l_p = self.ckpt_dir / f"best_model_fold_{self.fold_idx}.pth", self.ckpt_dir / f"latest_model_fold_{self.fold_idx}.pth"
        p = l_p if l_p.exists() else b_p
        if b_p.exists() and l_p.exists():
            c_b, c_l = torch.load(b_p, map_location="cpu"), torch.load(l_p, map_location="cpu")
            p = b_p if c_b.get("best_dsc", -1) > c_l.get("best_dsc", -1) else l_p
        if not p.exists(): return 1
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.best_dsc = ckpt.get("best_dsc", -1.0)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt: self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception: pass
        return int(ckpt["epoch"]) + 1

    def train_epoch(self, dataloader: Any) -> float:
        """Execute one training epoch with learning rate annealing."""
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            inputs = {k: batch[k].to(self.device) for k in ["raw", "lung", "medias"]}
            self.optimizer.zero_grad()
            loss, _ = self.criterion(self.model(inputs), batch["mask"].to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / max(1, len(dataloader))

    @torch.no_grad()
    def validate(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate model across validation set using volumetric accumulator."""
        self.model.eval()
        t_loss = 0.0
        acc = SegmentationMetricAccumulator(tau=self.config["evaluation"]["normalized_surface_distance_tau"])
        for batch in dataloader:
            inputs = {k: batch[k].to(self.device) for k in ["raw", "lung", "medias"]}
            masks = batch["mask"].to(self.device)
            outputs = self.model(inputs)
            loss, _ = self.criterion(outputs, masks)
            t_loss += loss.item()
            acc.update(masks, outputs["final_prediction"])
        m = acc.compute()
        return {"loss": t_loss / max(1, len(dataloader)), "dsc": m["global_dsc"], "iou": m["global_iou"], "nsd": m["nsd"]}

    def save_checkpoint(self, val_dsc: float, epoch: int) -> None:
        """Persist latest and best checkpoints directly to Drive."""
        state = {
            "epoch": epoch, "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": self.scheduler.state_dict(),
            "best_dsc": max(self.best_dsc, val_dsc), "val_dsc": val_dsc,
        }
        torch.save(state, self.ckpt_dir / f"latest_model_fold_{self.fold_idx}.pth")
        if val_dsc > self.best_dsc:
            self.best_dsc = val_dsc
            torch.save(state, self.ckpt_dir / f"best_model_fold_{self.fold_idx}.pth")

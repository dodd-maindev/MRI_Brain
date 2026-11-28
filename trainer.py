"""Trainer class with AdamW, linear warmup, gradient clipping, EMA, and Drive resume."""

from contextlib import nullcontext
import csv, math
from pathlib import Path
from typing import Any, Dict
import torch
import torch.optim as optim
from mus_net.losses import MultiWindowSegmentationLoss
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.utils.ema import ModelExponentialMovingAverage


class MusNetTrainer:
    """Encapsulates training, EMA validation, real-time logging, and resume logic."""

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], fold_idx: int) -> None:
        """Initialize optimizer, warmup-cosine scheduler, EMA, loss, and logs."""
        self.config, self.fold_idx = config, fold_idx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        tc, lc = config["train"], config["loss"]
        lr, wd, opt = tc["learning_rate"], tc.get("weight_decay", 1e-4), tc.get("optimizer", "adam").lower()
        self.optimizer = (optim.SGD(self.model.parameters(), lr=lr, momentum=tc.get("momentum", 0.9), weight_decay=wd) if opt == "sgd"
                          else optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd) if opt == "adamw"
                          else optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd))
        epochs, warmup = tc["epochs"], int(tc.get("warmup_epochs", 0))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda ep: (ep + 1) / warmup if warmup > 0 and ep < warmup
            else 0.005 + 0.4975 * (1.0 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup))),
        )
        self.grad_clip, ema_decay = float(tc.get("grad_clip", 0.0)), float(tc.get("ema_decay", 0.0))
        self.ema = ModelExponentialMovingAverage(self.model, ema_decay) if ema_decay > 0.0 else None
        self.supervision = lc.get("supervision", "ann0")
        self.criterion = MultiWindowSegmentationLoss(
            lc["alpha_1"], lc["alpha_2"], lc.get("use_dice_loss", True), lc.get("dice_weight", 1.0),
            lc.get("bce_weight", 1.0), lc["use_posterior_kl"], lc["kl_weight"], lc.get("boundary_weight", 0.0),
            lc.get("tversky_weight", 0.0), lc.get("use_uncertainty_aux", False), lc.get("iou_weight", 0.0),
            lc.get("use_evidential", False), lc.get("edl_kl_weight", 0.05), self.supervision,
        )
        self.best_dsc, self.ckpt_dir = -1.0, Path(tc["checkpoint_directory"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(tc["log_directory"]) / f"log_fold_{fold_idx}.csv"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_dsc", "val_iou", "val_nsd"])

    def log_epoch(self, epoch: int, tr_loss: float, v_loss: float, dsc: float, iou: float, nsd: float) -> None:
        """Append metrics to CSV and flush to Drive immediately."""
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{v_loss:.4f}", f"{dsc:.4f}", f"{iou:.4f}", f"{nsd:.4f}"])
    def resume_if_available(self) -> int:
        """Restore model, scheduler progress, and EMA state from checkpoint."""
        b_p, l_p = self.ckpt_dir / f"best_model_fold_{self.fold_idx}.pth", self.ckpt_dir / f"latest_model_fold_{self.fold_idx}.pth"
        p = l_p if l_p.exists() else b_p
        if not p.exists(): return 1
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if self.ema is not None: self.ema = ModelExponentialMovingAverage(self.model, self.ema.decay)
        self.best_dsc, ep = ckpt.get("best_dsc", -1.0), int(ckpt.get("epoch", 0))
        for _ in range(ep): self.scheduler.step()
        return ep + 1
    def train_epoch(self, dataloader: Any, epoch: int = 1) -> float:
        """Execute one training epoch with KL annealing, grad clip, EMA, and LR step."""
        self.model.train(); total_loss, anneal = 0.0, min(1.0, float(epoch) / 10.0)
        for batch in dataloader:
            inputs = {k: batch[k].to(self.device) for k in ["raw", "lung", "medias"]}
            self.optimizer.zero_grad()
            target = batch["mask"].to(self.device) if self.supervision == "ann0" else batch
            loss, _ = self.criterion(self.model(inputs), target, anneal)
            loss.backward()
            if self.grad_clip > 0.0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.ema is not None: self.ema.update(self.model)
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / max(1, len(dataloader))
    @torch.no_grad()
    def validate(self, dataloader: Any) -> Dict[str, float]:
        """Evaluate model (using EMA weights if enabled) across validation set."""
        self.model.eval(); t_loss = 0.0
        acc = SegmentationMetricAccumulator(tau=self.config["evaluation"]["normalized_surface_distance_tau"])
        with (self.ema.average_parameters(self.model) if self.ema is not None else nullcontext()):
            for batch in dataloader:
                inputs, masks = {k: batch[k].to(self.device) for k in ["raw", "lung", "medias"]}, batch["mask"].to(self.device)
                outputs = self.model(inputs); t_loss += self.criterion(outputs, masks)[0].item()
                acc.update(masks, outputs["final_prediction"])
        m = acc.compute()
        return {"loss": t_loss / max(1, len(dataloader)), "dsc": m["global_dsc"], "iou": m["global_iou"], "nsd": m["nsd"]}

    def save_checkpoint(self, val_dsc: float, epoch: int) -> None:
        """Persist latest and best checkpoints (using EMA weights if active) to Drive."""
        with (self.ema.average_parameters(self.model) if self.ema is not None else nullcontext()):
            state = {"epoch": epoch, "model_state_dict": self.model.state_dict(), "best_dsc": max(self.best_dsc, val_dsc), "val_dsc": val_dsc}
            torch.save(state, self.ckpt_dir / f"latest_model_fold_{self.fold_idx}.pth")
            if val_dsc > self.best_dsc:
                self.best_dsc = val_dsc; torch.save(state, self.ckpt_dir / f"best_model_fold_{self.fold_idx}.pth")

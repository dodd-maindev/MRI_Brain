"""Diagnostic script to isolate data-scale vs architecture bottlenecks."""

from pathlib import Path
import sys, argparse, torch, yaml

_parent = str(Path(__file__).resolve().parent.parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.train import build_network_from_config


class CheckpointDiagnostic:
    """Evaluates branch masks, threshold sensitivity, and nodule-size breakdown."""

    def __init__(self, config: dict, ckpt_path: Path, fold: int) -> None:
        """Load model checkpoint and validation dataloader."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_network_from_config(config).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.epoch = ckpt.get("epoch", "?")
        _, self.val_loader = prepare_fold_dataloaders(config, fold)

    @staticmethod
    def _global_dsc(inter: float, gt_sum: float, pred_sum: float) -> float:
        """Compute global Dice percentage."""
        denom = gt_sum + pred_sum
        return 100.0 if denom == 0 else float(200.0 * inter / denom)

    @torch.no_grad()
    def run(self) -> None:
        """Collect validation predictions and print diagnostic tables."""
        preds, gts, branches = [], [], {"lung": [], "medias": [], "raw": []}
        for batch in self.val_loader:
            out = self.model({k: batch[k].to(self.device) for k in ["raw", "lung", "medias"]})
            preds.append(out["final_prediction"].cpu())
            gts.append(batch["mask"].cpu())
            for name in branches:
                if out["branch_masks"].get(name) is not None:
                    branches[name].append(out["branch_masks"][name].cpu())

        pred_all, gt_all = torch.cat(preds, dim=0), torch.cat(gts, dim=0)
        print(f"\n=== DIAGNOSTIC (Checkpoint Epoch {self.epoch}, N={len(gt_all)} slices) ===")
        self._report_thresholds(gt_all, pred_all)
        self._report_branches(gt_all, pred_all, branches)
        self._report_size_buckets(gt_all, pred_all)

    def _report_thresholds(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """Sweep probability threshold from 0.30 to 0.60."""
        print("\n1) Threshold Sweep (Final Prediction):")
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            acc = SegmentationMetricAccumulator(tau=2.0, threshold=thr)
            acc.update(gt, pred)
            m = acc.compute()
            print(f"  thr={thr:.2f} | Global DSC={m['global_dsc']*100:.2f}% | Macro DSC={m['macro_dsc']*100:.2f}% | IoU={m['global_iou']*100:.2f}% | NSD={m['nsd']*100:.2f}%")

    def _report_branches(self, gt: torch.Tensor, pred: torch.Tensor, branches: dict) -> None:
        """Compare UIM auxiliary branch DSCs against final FFM prediction."""
        print("\n2) Branch vs Final DSC (at threshold 0.50):")
        items = [("final_prediction", pred)] + [(f"branch_{k}", torch.cat(v, dim=0)) for k, v in branches.items() if v]
        for label, tensor in items:
            p, g = (tensor >= 0.5), (gt >= 0.5)
            dsc = self._global_dsc(float((p & g).sum()), float(g.sum()), float(p.sum()))
            print(f"  {label:<18s}: Global DSC = {dsc:.2f}% | mean_prob_in_gt = {float(tensor[g].mean()):.3f}")

    def _report_size_buckets(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        """Measure Global DSC grouped by ground-truth nodule slice area."""
        print("\n3) DSC by Nodule Slice Area (at threshold 0.50):")
        areas = (gt >= 0.5).sum(dim=(1, 2, 3))
        buckets = [("Tiny (<30 px)", areas < 30), ("Small (30-100 px)", (areas >= 30) & (areas < 100)),
                   ("Medium (100-300 px)", (areas >= 100) & (areas < 300)), ("Large (>=300 px)", areas >= 300)]
        for title, sel in buckets:
            if sel.sum() > 0:
                g, p = (gt[sel] >= 0.5), (pred[sel] >= 0.5)
                print(f"  {title:<20s} ({int(sel.sum()):4d} slices): Global DSC = {self._global_dsc(float((p & g).sum()), float(g.sum()), float(p.sum())):.2f}%")


def main() -> None:
    """CLI entry point for checkpoint diagnostics."""
    p = argparse.ArgumentParser(description="Diagnose trained MUS-Net checkpoint.")
    for arg, kwargs in [("--config", {"default": "configs/default.yaml"}), ("--data_dir", {"required": True}),
                        ("--drive_dir", {"required": True}), ("--fold", {"type": int, "default": 0}),
                        ("--min_annotators", {"type": int, "default": None})]:
        p.add_argument(arg, **kwargs)
    args = p.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["dataset"]["data_directory"] = args.data_dir
    if args.min_annotators is not None:
        config["dataset"]["min_annotators"] = args.min_annotators
    ckpt = Path(args.drive_dir) / "checkpoints" / f"best_model_fold_{args.fold}.pth"
    CheckpointDiagnostic(config, ckpt, args.fold).run()


if __name__ == "__main__":
    main()

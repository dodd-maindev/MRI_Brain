"""Step 2: Leave-one-out agreement benchmark on existing checkpoints (no retraining)."""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

_parent = str(Path(__file__).resolve().parents[2])
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from mus_net.analysis.annotation_utils import center_crop_64, get_filtered_fold_val_indices
from mus_net.analysis.model_vs_radiologists import _infer_fold_probs


def _global_dsc(gt_list: list[np.ndarray], pred_list: list[np.ndarray]) -> float:
    """Compute Global Volumetric DSC (%) across a list of 2D binary masks."""
    inter = sum(int(np.logical_and(g, p).sum()) for g, p in zip(gt_list, pred_list))
    denom = sum(int(g.sum()) + int(p.sum()) for g, p in zip(gt_list, pred_list))
    return 100.0 if denom == 0 else 200.0 * inter / denom


def run_leave_one_out(
    edl_cfg_p: Path, edl_dir: Path, v5_cfg_p: Path, v5_dir: Path,
    cache_p: Path, multi_p: Path,
) -> None:
    """Evaluate human LOO DSC, EDL-v1 LOO DSC, and AMU-Net v5 LOO DSC on n >= 3 slices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edl_cfg = yaml.safe_load(edl_cfg_p.read_text(encoding="utf-8"))
    v5_cfg = yaml.safe_load(v5_cfg_p.read_text(encoding="utf-8"))
    for c in (edl_cfg, v5_cfg):
        c["dataset"]["data_directory"] = str(cache_p.parent)
    fold_indices = get_filtered_fold_val_indices(cache_p, edl_cfg)

    with np.load(cache_p) as c, np.load(multi_p) as m:
        num_anns = c["num_annotators"]
        ann_masks = center_crop_64(m["annotator_masks"], int(edl_cfg["dataset"]["patch_size"]))

    print("\n[B2] LOO baseline (n >= 3):")
    print("| Fold | Human LOO DSC | EDL-v1 LOO DSC | v5 LOO DSC |")
    h_all, e_all, v_all = [], [], []
    for f in range(edl_cfg["dataset"]["num_folds"]):
        idx = fold_indices[f]
        edl_p = (_infer_fold_probs(edl_cfg, edl_dir / "checkpoints" / f"best_model_fold_{f}.pth", f, device) >= 0.5).astype(np.uint8)
        v5_p = (_infer_fold_probs(v5_cfg, v5_dir / "checkpoints" / f"best_model_fold_{f}.pth", f, device) >= 0.5).astype(np.uint8)

        gt_cons, h_preds, e_preds, v_preds = [], [], [], []
        for pos, ci in enumerate(idx):
            n = min(int(num_anns[ci]), 4)
            if n < 3:
                continue
            stack = ann_masks[ci, :n]
            total = stack.sum(axis=0)
            for j in range(n):
                cons_wo_j = ((total - stack[j]) >= 0.5 * (n - 1)).astype(np.uint8)
                gt_cons.append(cons_wo_j)
                h_preds.append(stack[j])
                e_preds.append(edl_p[pos])
                v_preds.append(v5_p[pos])

        hd, ed, vd = _global_dsc(gt_cons, h_preds), _global_dsc(gt_cons, e_preds), _global_dsc(gt_cons, v_preds)
        h_all.append(hd); e_all.append(ed); v_all.append(vd)
        print(f"| {f:4d} | {hd:13.2f} | {ed:14.2f} | {vd:10.2f} |")
    print(f"| Mean | {np.mean(h_all):13.2f} | {np.mean(e_all):14.2f} | {np.mean(v_all):10.2f} |")


def main() -> None:
    """CLI entry point for Step 2 leave-one-out benchmark."""
    p = argparse.ArgumentParser(description="Leave-one-out agreement benchmark.")
    p.add_argument("--edl_config", type=str, default="/content/mus_net/configs/edl_mus_net.yaml")
    p.add_argument("--edl_dir", type=str, default="/content/drive/MyDrive/edl_mus_net_experiments_v1")
    p.add_argument("--v5_config", type=str, default="/content/mus_net/configs/amu_net.yaml")
    p.add_argument("--v5_dir", type=str, default="/content/drive/MyDrive/amu_net_experiments_v5")
    p.add_argument("--cache_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz")
    p.add_argument("--multi_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_multi_annotator_cache.npz")
    a = p.parse_args()
    run_leave_one_out(Path(a.edl_config), Path(a.edl_dir), Path(a.v5_config), Path(a.v5_dir), Path(a.cache_path), Path(a.multi_path))


if __name__ == "__main__":
    main()

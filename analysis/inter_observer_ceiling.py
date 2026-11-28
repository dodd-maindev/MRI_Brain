"""Step 2: Measure inter-observer ceiling (ann_k vs ann0) and diameter-stratified DSC."""

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
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.utils.q1_metrics import compute_slice_hd95

DIAM_BINS = [("[3,6)", 3.0, 6.0), ("[6,10)", 6.0, 10.0), ("[10,20)", 10.0, 20.0), ("[20,30]", 20.0, 30.01)]


def _eval_pairs(gt_list: list[np.ndarray], pred_list: list[np.ndarray], tau: float) -> dict[str, float]:
    """Evaluate a list of 2D binary (gt, pred) pairs using SegmentationMetricAccumulator."""
    acc = SegmentationMetricAccumulator(tau=tau, threshold=0.5)
    gt_t = torch.from_numpy(np.stack(gt_list)[:, None].astype(np.float32))
    pr_t = torch.from_numpy(np.stack(pred_list)[:, None].astype(np.float32))
    acc.update(gt_t, pr_t)
    res = acc.compute()
    res["hd95"] = float(np.mean([compute_slice_hd95(g, p) for g, p in zip(gt_list, pred_list)]))
    return res


def run_ceiling_analysis(config_path: Path, cache_path: Path, multi_path: Path) -> None:
    """Compute per-fold inter-observer metrics, consensus(others), and diameter bins."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tau = float(config["evaluation"]["normalized_surface_distance_tau"])
    p_size = int(config["dataset"].get("patch_size", 64))
    fold_indices = get_filtered_fold_val_indices(cache_path, config)

    with np.load(cache_path) as c, np.load(multi_path) as m:
        diams, nkeys = c["diameter_mm"], c["nodule_keys"]
        ann_masks = center_crop_64(m["annotator_masks"], p_size)
        present = m["annotator_present"]

    print("\n[B2] Inter-observer (ann_k vs ann0), per fold + mean:")
    print("| Fold | DSC (%) | IoU (%) | NSD (%) | HD95 | Macro DSC (%) |")
    obs_rows, cons_rows = [], []
    bin_pairs: dict[str, tuple[list, list, set]] = {b[0]: ([], [], set()) for b in DIAM_BINS}

    for f_idx, val_idx in enumerate(fold_indices):
        gt_k, pr_k, gt_c, pr_c = [], [], [], []
        for idx in val_idx:
            g0 = ann_masks[idx, 0]
            others = [k for k in (1, 2, 3) if present[idx, k]]
            if not others:
                continue
            for k in others:
                gt_k.append(g0); pr_k.append(ann_masks[idx, k])
                for b_name, lo, hi in DIAM_BINS:
                    if lo <= float(diams[idx]) < hi:
                        bin_pairs[b_name][0].append(g0); bin_pairs[b_name][1].append(ann_masks[idx, k])
                        bin_pairs[b_name][2].add(str(nkeys[idx]))
            cons = (ann_masks[idx, others].sum(axis=0) >= 0.5 * len(others)).astype(np.uint8)
            gt_c.append(g0); pr_c.append(cons)

        r_k, r_c = _eval_pairs(gt_k, pr_k, tau), _eval_pairs(gt_c, pr_c, tau)
        obs_rows.append(r_k); cons_rows.append(r_c)
        print(f"| {f_idx:4d} | {r_k['global_dsc']*100:7.2f} | {r_k['global_iou']*100:7.2f} | {r_k['nsd']*100:7.2f} | {r_k['hd95']:4.2f} | {r_k['macro_dsc']*100:13.2f} |")

    print(f"| Mean | {np.mean([r['global_dsc'] for r in obs_rows])*100:7.2f} | {np.mean([r['global_iou'] for r in obs_rows])*100:7.2f} | "
          f"{np.mean([r['nsd'] for r in obs_rows])*100:7.2f} | {np.mean([r['hd95'] for r in obs_rows]):4.2f} | {np.mean([r['macro_dsc'] for r in obs_rows])*100:13.2f} |")

    print("\n[B2] Consensus(others) vs ann0:\n| Fold | DSC (%) | IoU (%) | NSD (%) |")
    for f_idx, r in enumerate(cons_rows):
        print(f"| {f_idx:4d} | {r['global_dsc']*100:7.2f} | {r['global_iou']*100:7.2f} | {r['nsd']*100:7.2f} |")
    print(f"| Mean | {np.mean([r['global_dsc'] for r in cons_rows])*100:7.2f} | {np.mean([r['global_iou'] for r in cons_rows])*100:7.2f} | {np.mean([r['nsd'] for r in cons_rows])*100:7.2f} |")

    print("\n[B2] Stratified ann_k vs ann0:\n| Bin | n_nodules | DSC ann_k vs ann0 |")
    for b_name, _, _ in DIAM_BINS:
        gl, pl, nset = bin_pairs[b_name]
        dsc = _eval_pairs(gl, pl, tau)["global_dsc"] * 100 if gl else 0.0
        print(f"| {b_name:7s} | {len(nset):9d} | {dsc:17.2f} |")


def main() -> None:
    """CLI entry point for Step 2 inter-observer ceiling."""
    p = argparse.ArgumentParser(description="Measure LIDC inter-observer ceiling.")
    p.add_argument("--config", type=str, default="/content/mus_net/configs/edl_mus_net.yaml")
    p.add_argument("--cache_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz")
    p.add_argument("--multi_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_multi_annotator_cache.npz")
    args = p.parse_args()
    run_ceiling_analysis(Path(args.config), Path(args.cache_path), Path(args.multi_path))


if __name__ == "__main__":
    main()

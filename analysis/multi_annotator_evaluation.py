"""Step 6/7: Evaluate B0, B1, B2, M on ann0, LOO, soft calibration, and uncertainty AUROCs."""

import argparse, sys
from pathlib import Path
import numpy as np, torch, yaml

_parent = str(Path(__file__).resolve().parents[2])
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from mus_net.analysis.annotation_utils import center_crop_64, get_filtered_fold_val_indices
from mus_net.analysis.multi_eval_metrics import (
    compute_loo_dsc, compute_soft_calibration, compute_union_roi_mask, partial_spearman, safe_auroc,
)
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.train import build_network_from_config
from mus_net.utils.uncertainty_decomposition import decompose_beta_uncertainty


@torch.no_grad()
def _eval_fold(cfg: dict, ckpt_p: Path, fold: int, ann_m: np.ndarray, n_anns: np.ndarray, idx: list[int], dev: torch.device) -> dict:
    """Run single-pass evaluation on one fold for all multi-annotator metrics."""
    model = build_network_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(ckpt_p, map_location=dev)["model_state_dict"], strict=False)
    model.eval()
    _, val_loader = prepare_fold_dataloaders(cfg, fold)
    probs, alea, epi, vac, clm = [], [], [], [], []
    for b in val_loader:
        out = model({k: b[k].to(dev) for k in ["raw", "lung", "medias"]})
        probs.append(out["final_prediction"][:, 0].cpu().numpy())
        a_f, b_f = out["final_evidence"]
        dec = decompose_beta_uncertainty(a_f[:, 0], b_f[:, 0])
        alea.append(dec["aleatoric"].cpu().numpy()); epi.append(dec["epistemic"].cpu().numpy())
        vac.append(dec["vacuity"].cpu().numpy()); clm.append(out["conflict_map"][:, 0].cpu().numpy())

    p = np.concatenate(probs); al = np.concatenate(alea); ep = np.concatenate(epi); cl = np.concatenate(clm)
    tau = float(cfg["evaluation"]["normalized_surface_distance_tau"])
    acc = SegmentationMetricAccumulator(tau=tau, threshold=0.5)
    acc.update(torch.from_numpy(ann_m[idx, 0:1].astype(np.float32)), torch.from_numpy(p[:, None].astype(np.float32)))
    m0 = acc.compute()

    n_slice = np.minimum(n_anns[idx], 4)
    y_pos = np.array([ann_m[ci, :n].sum(axis=0) for ci, n in zip(idx, n_slice)], dtype=np.float32)
    y_bar = y_pos / n_slice[:, None, None]
    cal = compute_soft_calibration(p, y_bar)
    loo = compute_loo_dsc((p >= 0.5).astype(np.uint8), ann_m[idx], n_anns[idx])

    roi = compute_union_roi_mask(ann_m[idx], n_anns[idx], dilate_px=3)
    disagree = ((y_pos > 0) & (y_pos < n_slice[:, None, None]))[roi].astype(np.uint8)
    err = ((p >= 0.5) != (y_bar >= 0.5))[roi].astype(np.uint8)
    return {
        "dsc": m0["global_dsc"] * 100, "iou": m0["global_iou"] * 100, "nsd": m0["nsd"] * 100,
        "loo": loo, "brier": cal["brier"], "ece": cal["ece"],
        "auroc_alea_dis": safe_auroc(al[roi], disagree), "auroc_epi_err": safe_auroc(ep[roi], err),
        "auroc_clm_dis": safe_auroc(cl[roi], disagree), "part_r": partial_spearman(ep[roi], err, al[roi]),
    }


def main() -> None:
    """CLI entry point for evaluating B0, B1, B2, and M checkpoints."""
    ap = argparse.ArgumentParser(description="Evaluate multi-annotator EDL models.")
    for k, kw in [("--folds", {"type": int, "nargs": "+", "default": [1]}), ("--ckpt_kinds", {"nargs": "+", "default": ["best", "latest"]}),
                  ("--cache_path", {"default": "/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz"}),
                  ("--multi_path", {"default": "/content/drive/MyDrive/lidc_full_v2/lidc_multi_annotator_cache.npz"})]:
        ap.add_argument(k, **kw)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = [("B0", "/content/mus_net/configs/edl_mus_net.yaml", "/content/drive/MyDrive/edl_mus_net_experiments_v1"),
               ("B1", "/content/mus_net/configs/edl_random_annotator.yaml", "/content/drive/MyDrive/edl_b1_random_v1"),
               ("B2", "/content/mus_net/configs/edl_soft_label.yaml", "/content/drive/MyDrive/edl_b2_soft_v1"),
               ("M", "/content/mus_net/configs/edl_beta_binomial.yaml", "/content/drive/MyDrive/edl_m_betabinomial_v1")]
    base_cfg = yaml.safe_load(Path(methods[0][1]).read_text(encoding="utf-8"))
    base_cfg["dataset"]["data_directory"] = str(Path(args.cache_path).parent)
    fold_indices = get_filtered_fold_val_indices(Path(args.cache_path), base_cfg)
    with np.load(args.cache_path) as c, np.load(args.multi_path) as m:
        n_anns = c["num_annotators"]
        ann_m = center_crop_64(m["annotator_masks"], int(base_cfg["dataset"]["patch_size"]))

    for kind in args.ckpt_kinds:
        print(f"\n[B6] Evaluation ({kind} checkpoint):")
        print("| Method | DSC_ann0 | IoU | NSD | LOO DSC | Brier(ybar) | ECE(ybar) | AUROC alea->dis | AUROC epi->err | AUROC C_lm->dis | part_r |")
        for name, cfg_p, d_dir in methods:
            cfg = yaml.safe_load(Path(cfg_p).read_text(encoding="utf-8"))
            cfg["dataset"]["data_directory"] = str(Path(args.cache_path).parent)
            rows = []
            for f in args.folds:
                ckpt = Path(d_dir) / "checkpoints" / f"{kind}_model_fold_{f}.pth"
                if ckpt.exists():
                    rows.append(_eval_fold(cfg, ckpt, f, ann_m, n_anns, fold_indices[f], dev))
            if not rows:
                continue
            avg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
            print(f"| {name:6s} | {avg['dsc']:8.2f} | {avg['iou']:5.2f} | {avg['nsd']:5.2f} | {avg['loo']:7.2f} | "
                  f"{avg['brier']:11.4f} | {avg['ece']:9.4f} | {avg['auroc_alea_dis']:15.4f} | {avg['auroc_epi_err']:14.4f} | "
                  f"{avg['auroc_clm_dis']:15.4f} | {avg['part_r']:6.4f} |")


if __name__ == "__main__":
    main()

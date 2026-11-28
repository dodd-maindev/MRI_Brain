"""Step 3: Single-pass evaluation of EDL-MUS-Net and AMU-Net v5 vs individual radiologists."""

import argparse, sys
from pathlib import Path
import numpy as np, torch, yaml

_parent = str(Path(__file__).resolve().parents[2])
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from mus_net.analysis.annotation_utils import center_crop_64, get_filtered_fold_val_indices
from mus_net.data.data_factory import prepare_fold_dataloaders
from mus_net.metrics import SegmentationMetricAccumulator
from mus_net.train import build_network_from_config


@torch.no_grad()
def _infer_fold_probs(config: dict, ckpt_path: Path, fold: int, device: torch.device) -> np.ndarray:
    """Run single-pass (no TTA/LCC) inference and return [N, H, W] probabilities."""
    model = build_network_from_config(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.eval()
    _, val_loader = prepare_fold_dataloaders(config, fold)
    probs = [model({k: b[k].to(device) for k in ["raw", "lung", "medias"]})["final_prediction"][:, 0].cpu().numpy() for b in val_loader]
    return np.concatenate(probs, axis=0)


def _score_vs_masks(probs: np.ndarray, gts: np.ndarray, tau: float) -> tuple[float, float, float]:
    """Compute (Global DSC, Global IoU, Global NSD) in percent for [N, H, W] arrays."""
    acc = SegmentationMetricAccumulator(tau=tau, threshold=0.5)
    acc.update(torch.from_numpy(gts[:, None].astype(np.float32)), torch.from_numpy(probs[:, None].astype(np.float32)))
    r = acc.compute()
    return r["global_dsc"] * 100, r["global_iou"] * 100, r["nsd"] * 100


def evaluate_models(
    edl_cfg_p: Path, edl_dir: Path, v5_cfg_p: Path, v5_dir: Path,
    cache_p: Path, multi_p: Path, out_p: Path,
) -> None:
    """Compare EDL-MUS-Net and AMU-Net v5 against ann0, ann_k, and consensus(4)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edl_cfg = yaml.safe_load(edl_cfg_p.read_text(encoding="utf-8"))
    v5_cfg = yaml.safe_load(v5_cfg_p.read_text(encoding="utf-8"))
    for c, d in [(edl_cfg, cache_p.parent), (v5_cfg, cache_p.parent)]:
        c["dataset"]["data_directory"] = str(d)
    tau = float(edl_cfg["evaluation"]["normalized_surface_distance_tau"])
    fold_indices = get_filtered_fold_val_indices(cache_p, edl_cfg)

    with np.load(cache_p) as c, np.load(multi_p) as m:
        pids, nkeys, diams = c["patient_ids"], c["nodule_keys"], c["diameter_mm"]
        ann_masks = center_crop_64(m["annotator_masks"], int(edl_cfg["dataset"]["patch_size"]))
        present = m["annotator_present"]

    print("\n[B3] | Fold | Model | vs ann0 (DSC/IoU/NSD) | vs ann_k (DSC) | vs consensus(4) (DSC) |")
    records = []
    for name, cfg, ckpt_root in [("EDL", edl_cfg, edl_dir), ("v5", v5_cfg, v5_dir)]:
        s0, si, sn, sk, sc = [], [], [], [], []
        for f in range(cfg["dataset"]["num_folds"]):
            probs = _infer_fold_probs(cfg, ckpt_root / "checkpoints" / f"best_model_fold_{f}.pth", f, device)
            idx = fold_indices[f]
            assert len(probs) == len(idx), f"Val size mismatch fold {f}: {len(probs)} vs {len(idx)}"
            d0, i0, n0 = _score_vs_masks(probs, ann_masks[idx, 0], tau)
            pk = [probs[j] for j, ci in enumerate(idx) for k in (1, 2, 3) if present[ci, k]]
            gk = [ann_masks[ci, k] for ci in idx for k in (1, 2, 3) if present[ci, k]]
            dk = _score_vs_masks(np.stack(pk), np.stack(gk), tau)[0]
            cons4 = (ann_masks[idx].sum(axis=1) >= 0.5 * present[idx].sum(axis=1)[:, None, None]).astype(np.uint8)
            dc4 = _score_vs_masks(probs, cons4, tau)[0]
            s0.append(d0); si.append(i0); sn.append(n0); sk.append(dk); sc.append(dc4)
            print(f"| {f:4d} | {name:5s} | {d0:5.2f} / {i0:5.2f} / {n0:5.2f} | {dk:14.2f} | {dc4:21.2f} |")
            pred_b, gt0 = (probs >= 0.5).astype(np.uint8), ann_masks[idx, 0]
            for j, ci in enumerate(idx):
                inter = int(np.logical_and(pred_b[j], gt0[j]).sum())
                records.append((name, f, str(pids[ci]), str(nkeys[ci]), float(diams[ci]), inter, int(gt0[j].sum()), int(pred_b[j].sum())))
        print(f"| Mean | {name:5s} | {np.mean(s0):5.2f} / {np.mean(si):5.2f} / {np.mean(sn):5.2f} | {np.mean(sk):14.2f} | {np.mean(sc):21.2f} |")

    arr = np.array(records, dtype=object)
    np.savez(out_p, records=arr)
    print(f"Saved slice stats for Step 4 to {out_p}")


def main() -> None:
    """CLI entry point for Step 3 model-vs-radiologists evaluation."""
    p = argparse.ArgumentParser(description="Evaluate EDL and v5 vs radiologists.")
    p.add_argument("--edl_config", type=str, default="/content/mus_net/configs/edl_mus_net.yaml")
    p.add_argument("--edl_dir", type=str, default="/content/drive/MyDrive/edl_mus_net_experiments_v1")
    p.add_argument("--v5_config", type=str, default="/content/mus_net/configs/amu_net.yaml")
    p.add_argument("--v5_dir", type=str, default="/content/drive/MyDrive/amu_net_experiments_v5")
    p.add_argument("--cache_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz")
    p.add_argument("--multi_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_multi_annotator_cache.npz")
    p.add_argument("--out_path", type=str, default="/content/drive/MyDrive/edl_mus_net_experiments_v1/analysis_slice_stats.npz")
    a = p.parse_args()
    evaluate_models(Path(a.edl_config), Path(a.edl_dir), Path(a.v5_config), Path(a.v5_dir), Path(a.cache_path), Path(a.multi_path), Path(a.out_path))


if __name__ == "__main__":
    main()

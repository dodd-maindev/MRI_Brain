"""Step 4: Per-nodule paired Wilcoxon/t-test, Welch t-test vs paper, and patient bootstrap CI."""

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

DIAM_BINS = [("[3,6)", 3.0, 6.0), ("[6,10)", 6.0, 10.0), ("[10,20)", 10.0, 20.0), ("[20,30]", 20.0, 30.01)]
PAPER_STATS = {"DSC": (87.34, 0.62, 5), "IoU": (78.31, 0.89, 5), "NSD": (96.06, 0.61, 5)}


def _aggregate_by_key(rows: np.ndarray, key_col: int) -> dict[str, float]:
    """Aggregate (inter, gt, pred) by nodule_key or patient_id and compute DSC (%)."""
    sums: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        s = sums[str(r[key_col])]
        s[0] += int(r[5]); s[1] += int(r[6]); s[2] += int(r[7])
    return {k: (100.0 if (v[1] + v[2]) == 0 else 200.0 * v[0] / (v[1] + v[2])) for k, v in sums.items()}


def run_significance_tests(stats_path: Path, edl_folds: list[float], iou_folds: list[float], nsd_folds: list[float]) -> None:
    """Compute paired tests (EDL vs v5), Welch t-tests vs paper, stratified DSC, and bootstrap CI."""
    records = np.load(stats_path, allow_pickle=True)["records"]
    edl_rows = records[records[:, 0] == "EDL"]
    v5_rows = records[records[:, 0] == "v5"]

    edl_nod = _aggregate_by_key(edl_rows, 3)
    v5_nod = _aggregate_by_key(v5_rows, 3)
    common = sorted(set(edl_nod) & set(v5_nod))
    d_edl = np.array([edl_nod[k] for k in common])
    d_v5 = np.array([v5_nod[k] for k in common])
    diff = d_edl - d_v5
    w_stat, w_p = stats.wilcoxon(d_edl, d_v5)
    t_stat, t_p = stats.ttest_rel(d_edl, d_v5)
    print(f"\n[B4] EDL vs v5 per-nodule (N={len(common)}): mean diff={diff.mean():+.4f}%, Wilcoxon p={w_p:.4g}, t-test p={t_p:.4g}")

    for name, vals in [("DSC", edl_folds), ("IoU", iou_folds), ("NSD", nsd_folds)]:
        pm, ps, pn = PAPER_STATS[name]
        wt, wp = stats.ttest_ind_from_stats(
            float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals), pm, ps, pn, equal_var=False
        )
        print(f"[B4] Welch vs paper ({name}): mean={np.mean(vals):.2f}+/-{np.std(vals, ddof=1):.2f}, t={wt:.4f}, p={wp:.4g}")

    pat_sums: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in edl_rows:
        s = pat_sums[str(r[2])]
        s[0] += int(r[5]); s[1] += int(r[6]); s[2] += int(r[7])
    pat_arr = np.array(list(pat_sums.values()), dtype=np.float64)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(pat_arr), size=len(pat_arr))
        tot = pat_arr[idx].sum(axis=0)
        boot.append(200.0 * tot[0] / (tot[1] + tot[2]))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(f"[B4] Bootstrap 95% CI Global DSC (EDL, 2000 patient resamples): [{ci_lo:.2f}%, {ci_hi:.2f}%]")

    print("\n[B4] Stratified EDL vs ann0:")
    for b_name, lo, hi in DIAM_BINS:
        sub = edl_rows[(edl_rows[:, 4].astype(float) >= lo) & (edl_rows[:, 4].astype(float) < hi)]
        inter, gt, pr = int(sub[:, 5].sum()), int(sub[:, 6].sum()), int(sub[:, 7].sum())
        dsc = 200.0 * inter / (gt + pr) if (gt + pr) > 0 else 0.0
        print(f"| {b_name:7s} | n_nodules={len(set(sub[:, 3])):4d} | DSC EDL vs ann0={dsc:.2f}% |")


def main() -> None:
    """CLI entry point for Step 4 statistical significance tests."""
    p = argparse.ArgumentParser(description="Run Step 4 statistical tests.")
    p.add_argument("--stats_path", type=str, default="/content/drive/MyDrive/edl_mus_net_experiments_v1/analysis_slice_stats.npz")
    p.add_argument("--edl_dsc", type=float, nargs=5, default=[88.42, 87.61, 86.06, 87.34, 89.69])
    p.add_argument("--edl_iou", type=float, nargs=5, default=[79.24, 77.96, 75.53, 77.52, 81.30])
    p.add_argument("--edl_nsd", type=float, nargs=5, default=[97.44, 95.94, 95.95, 95.66, 97.33])
    a = p.parse_args()
    run_significance_tests(Path(a.stats_path), a.edl_dsc, a.edl_iou, a.edl_nsd)


if __name__ == "__main__":
    main()

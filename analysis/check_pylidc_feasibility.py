"""Step 0: Check pylidc boolean_mask() feasibility without DICOM and verify ann0 crop IoU."""

import argparse
from pathlib import Path
import sys
import numpy as np
import pylidc as pl

_parent = str(Path(__file__).resolve().parents[2])
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from mus_net.analysis.annotation_utils import (
    ensure_pylidc_numpy_compat, extract_aligned_slice_mask, parse_nodule_key,
)


def run_feasibility_check(cache_path: Path, num_slices: int = 50, seed: int = 42) -> None:
    """Test pylidc boolean_mask() without DICOM and measure ann0 reconstruction IoU."""
    ensure_pylidc_numpy_compat()
    with np.load(cache_path) as cache:
        masks = cache["masks"]
        nodule_keys = cache["nodule_keys"]
        slice_indices = cache["slice_indices"]

    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(len(masks), size=min(num_slices, len(masks)), replace=False)
    scan_cache, ious = {}, []

    try:
        for idx in sample_ids:
            scan_id, nodule_idx = parse_nodule_key(str(nodule_keys[idx]))
            if scan_id not in scan_cache:
                scan_cache[scan_id] = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).one()
            ann0 = scan_cache[scan_id].cluster_annotations()[nodule_idx][0]
            bbox = ann0.bbox()
            cy, cx = (bbox[0].start + bbox[0].stop) // 2, (bbox[1].start + bbox[1].stop) // 2
            recon = extract_aligned_slice_mask(
                ann0, int(slice_indices[idx]), cy, cx, patch_size=masks.shape[1]
            )
            gt = (masks[idx] > 0).astype(np.uint8)
            if recon is None:
                ious.append(0.0)
            else:
                inter = np.logical_and(recon > 0, gt > 0).sum()
                union = np.logical_or(recon > 0, gt > 0).sum()
                ious.append(1.0 if union == 0 else float(inter / union))
        print("[B0] boolean_mask without DICOM: OK")
        print(f"[B0] ann0 reconstruction IoU on {len(ious)} slices: min={min(ious):.6f}, mean={np.mean(ious):.6f}")
    except Exception as exc:
        print(f"[B0] boolean_mask without DICOM: FAIL ({type(exc).__name__}: {exc})")
        raise


def main() -> None:
    """CLI entry point for Step 0 feasibility check."""
    parser = argparse.ArgumentParser(description="Step 0 pylidc feasibility check.")
    parser.add_argument("--cache_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz")
    parser.add_argument("--num_slices", type=int, default=50)
    args = parser.parse_args()
    run_feasibility_check(Path(args.cache_path), args.num_slices)


if __name__ == "__main__":
    main()

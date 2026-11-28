"""Step 1: Build multi-annotator mask cache aligned 1:1 with lidc_packed_cache_25d.npz."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import pylidc as pl

_parent = str(Path(__file__).resolve().parents[2])
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from mus_net.analysis.annotation_utils import extract_aligned_slice_mask, parse_nodule_key


def build_cache(cache_path: Path, output_path: Path) -> None:
    """Reconstruct up to 4 radiologist masks per slice aligned to ann0 crop center."""
    with np.load(cache_path) as cache:
        masks = cache["masks"]
        nodule_keys = cache["nodule_keys"]
        slice_indices = cache["slice_indices"]

    n_slices, h, w = masks.shape
    annotator_masks = np.zeros((n_slices, 4, h, w), dtype=np.uint8)
    annotator_present = np.zeros((n_slices, 4), dtype=bool)

    by_scan = defaultdict(list)
    for idx in range(n_slices):
        scan_id, nodule_idx = parse_nodule_key(str(nodule_keys[idx]))
        by_scan[scan_id].append((idx, nodule_idx, int(slice_indices[idx])))

    scan_ids = sorted(by_scan.keys())
    for s_num, scan_id in enumerate(scan_ids, 1):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).one()
        clusters = scan.cluster_annotations()
        for idx, nodule_idx, abs_z in by_scan[scan_id]:
            anns = clusters[nodule_idx][:4]
            bbox0 = anns[0].bbox()
            cy, cx = (bbox0[0].start + bbox0[0].stop) // 2, (bbox0[1].start + bbox0[1].stop) // 2
            for k, ann in enumerate(anns):
                m = extract_aligned_slice_mask(ann, abs_z, cy, cx, patch_size=h)
                if m is not None:
                    annotator_masks[idx, k] = m
                    annotator_present[idx, k] = True
        if s_num % 100 == 0 or s_num == len(scan_ids):
            print(f"Processed {s_num}/{len(scan_ids)} scans...", flush=True)

    assert np.array_equal(annotator_masks[:, 0], (masks > 0).astype(np.uint8)), "ann0 mismatch!"
    counts = annotator_present.sum(axis=1)
    dist = {int(k): int((counts == k).sum()) for k in sorted(np.unique(counts))}
    np.savez_compressed(output_path, annotator_masks=annotator_masks, annotator_present=annotator_present)
    print(f"[B1] N slices={n_slices}, annotators present distribution: {dist}")
    print(f"Saved multi-annotator cache to {output_path}")


def main() -> None:
    """CLI entry point for building multi-annotator cache."""
    parser = argparse.ArgumentParser(description="Build multi-annotator LIDC mask cache.")
    parser.add_argument("--cache_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_packed_cache_25d.npz")
    parser.add_argument("--output_path", type=str, default="/content/drive/MyDrive/lidc_full_v2/lidc_multi_annotator_cache.npz")
    args = parser.parse_args()
    build_cache(Path(args.cache_path), Path(args.output_path))


if __name__ == "__main__":
    main()

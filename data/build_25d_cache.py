"""Build 2.5D packed LIDC cache with nodule keys and axial slice indices."""

import argparse
import os
import re
from pathlib import Path
import numpy as np

CACHE_25D_FILENAME = "lidc_packed_cache_25d.npz"
SLICE_PATTERN = re.compile(r"^(.*)_slice(\d+)\.npz$")


def build_25d_cache(data_directory: Path) -> None:
    """Pack all per-slice .npz files into a single 2.5D-indexed compressed cache."""
    with os.scandir(data_directory) as entries:
        names = sorted(
            e.name for e in entries
            if e.name.endswith(".npz") and not e.name.startswith("lidc_packed_cache")
        )
    if not names:
        raise FileNotFoundError(f"No individual .npz slices found in {data_directory}")

    hus, masks, pids, anns, diams, nkeys, zidxs = [], [], [], [], [], [], []
    for idx, name in enumerate(names, 1):
        if idx % 2000 == 0 or idx == len(names):
            print(f"Packing 2.5D cache: {idx}/{len(names)} slices...", flush=True)
        match = SLICE_PATTERN.match(name)
        nkey = match.group(1) if match else name
        zidx = int(match.group(2)) if match else 0
        with np.load(data_directory / name) as data:
            hus.append(data["hu_patch"])
            masks.append(data["mask"])
            pids.append(str(data["patient_id"]) if "patient_id" in data else name.split("_")[0])
            anns.append(int(data["num_annotators"]) if "num_annotators" in data else -1)
            diams.append(float(data["diameter_mm"]) if "diameter_mm" in data else -1.0)
            nkeys.append(str(data["nodule_key"]) if "nodule_key" in data else nkey)
            zidxs.append(zidx)

    out_path = data_directory / CACHE_25D_FILENAME
    print(f"Writing {out_path}...", flush=True)
    np.savez_compressed(
        out_path,
        hu_patches=np.stack(hus),
        masks=np.stack(masks),
        patient_ids=np.array(pids),
        num_annotators=np.array(anns, dtype=np.int16),
        diameter_mm=np.array(diams, dtype=np.float32),
        nodule_keys=np.array(nkeys),
        slice_indices=np.array(zidxs, dtype=np.int32),
    )
    print(f"Done! Saved {len(names)} slices to {out_path}.", flush=True)


def main() -> None:
    """CLI entry point for building 2.5D packed cache."""
    parser = argparse.ArgumentParser(description="Build 2.5D LIDC packed cache.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with .npz slices")
    args = parser.parse_args()
    build_25d_cache(Path(args.data_dir))


if __name__ == "__main__":
    main()

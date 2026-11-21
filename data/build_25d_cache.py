"""Build 2.5D packed LIDC cache with nodule keys and axial slice indices."""

import argparse, os, re
from pathlib import Path
import numpy as np

CACHE_FILENAME = "lidc_packed_cache.npz"
CACHE_25D_FILENAME = "lidc_packed_cache_25d.npz"
SLICE_PATTERN = re.compile(r"^(.*)_slice(\d+)\.npz$")


def build_25d_cache(data_directory: Path) -> None:
    """Pack LIDC slices into a 2.5D-indexed cache (reusing existing packed cache if present)."""
    with os.scandir(data_directory) as entries:
        names = sorted(e.name for e in entries if e.name.endswith(".npz") and not e.name.startswith("lidc_packed_cache"))
    if not names:
        raise FileNotFoundError(f"No individual .npz slices found in {data_directory}")

    matches = [SLICE_PATTERN.match(n) for n in names]
    nkeys = [m.group(1) if m else n for m, n in zip(matches, names)]
    zidxs = [int(m.group(2)) if m else i for i, m in enumerate(matches)]

    base_cache = data_directory / CACHE_FILENAME
    if base_cache.exists():
        with np.load(base_cache) as c:
            if len(c["hu_patches"]) == len(names):
                print(f"Upgrading {CACHE_FILENAME} -> {CACHE_25D_FILENAME} using filename index...", flush=True)
                hus, masks, pids, anns, diams = (
                    c["hu_patches"], c["masks"], c["patient_ids"], c["num_annotators"], c["diameter_mm"]
                )
            else:
                hus = None
    else:
        hus = None

    if hus is None:
        hus, masks, pids, anns, diams = [], [], [], [], []
        for idx, name in enumerate(names, 1):
            if idx % 2000 == 0 or idx == len(names):
                print(f"Packing 2.5D cache: {idx}/{len(names)} slices...", flush=True)
            with np.load(data_directory / name) as data:
                hus.append(data["hu_patch"]); masks.append(data["mask"])
                pids.append(str(data["patient_id"]) if "patient_id" in data else name.split("_")[0])
                anns.append(int(data["num_annotators"]) if "num_annotators" in data else -1)
                diams.append(float(data["diameter_mm"]) if "diameter_mm" in data else -1.0)
        hus, masks = np.stack(hus), np.stack(masks)
        pids, anns, diams = np.array(pids), np.array(anns, dtype=np.int16), np.array(diams, dtype=np.float32)

    out_path = data_directory / CACHE_25D_FILENAME
    np.savez_compressed(
        out_path, hu_patches=hus, masks=masks, patient_ids=pids, num_annotators=anns,
        diameter_mm=diams, nodule_keys=np.array(nkeys), slice_indices=np.array(zidxs, dtype=np.int32),
    )
    print(f"Done! Saved {len(names)} slices to {out_path}.", flush=True)


def main() -> None:
    """CLI entry point for building 2.5D packed cache."""
    parser = argparse.ArgumentParser(description="Build 2.5D LIDC packed cache.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with .npz slices")
    build_25d_cache(Path(parser.parse_args().data_dir))


if __name__ == "__main__":
    main()

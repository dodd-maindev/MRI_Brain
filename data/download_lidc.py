"""Helper to download and verify preprocessed LIDC-IDRI nodule dataset."""

import argparse
from pathlib import Path
import subprocess
import numpy as np


def verify_preprocessed_dataset(dataset_dir: str) -> None:
    """Scan preprocessed directory and display slice and patient statistics."""
    p = Path(dataset_dir)
    npz_files = list(p.glob("*.npz"))
    print(f"Total .npz slices found: {len(npz_files)}")
    if not npz_files:
        print("Warning: Directory contains no .npz files.")
        return

    patients = set()
    for f in npz_files[:100]:
        data = np.load(f)
        patients.add(str(data.get("patient_id", f.stem.split("_")[0])))
    print(f"Sample inspected: {len(patients)} unique patients in first 100 slices.")
    first_data = np.load(npz_files[0])
    print(f"Patch shape: {first_data['hu_patch'].shape}, Mask shape: {first_data['mask'].shape}")


def download_from_kaggle(dataset_slug: str, output_dir: str) -> None:
    """Download and extract preprocessed LIDC dataset using Kaggle CLI."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset_slug} -p {output_dir} --unzip"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    verify_preprocessed_dataset(output_dir)


def main() -> None:
    """CLI for downloading and verifying LIDC dataset."""
    parser = argparse.ArgumentParser(description="Download or verify LIDC-IDRI data.")
    parser.add_argument("--kaggle_dataset", type=str, default=None, help="Kaggle dataset slug")
    parser.add_argument("--output_dir", type=str, default="data/lidc_preprocessed")
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    if args.kaggle_dataset:
        download_from_kaggle(args.kaggle_dataset, args.output_dir)
    else:
        verify_preprocessed_dataset(args.output_dir)


if __name__ == "__main__":
    main()


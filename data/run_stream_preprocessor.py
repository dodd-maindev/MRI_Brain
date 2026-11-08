"""CLI runner for streaming LIDC-IDRI download and extraction."""

import mus_net.data.numpy_compatibility  # noqa: F401
import argparse
from pathlib import Path
import shutil
from mus_net.data.stream_preprocessor import StreamingLidcPreprocessor


def main() -> None:
    """Execute streaming preprocessor across patients."""
    parser = argparse.ArgumentParser(description="Streaming LIDC download and crop.")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/lidc_preprocessed")
    parser.add_argument("--temp_dir", type=str, default="/content/temp_dicom")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()

    out_p, temp_p = Path(args.output_dir), Path(args.temp_dir)
    preprocessor = StreamingLidcPreprocessor(out_p, temp_p, args.patch_size)

    patient_ids = preprocessor.downloader.fetch_all_patient_ids()
    if args.start_index > 0:
        patient_ids = patient_ids[args.start_index :]
    if args.max_patients is not None:
        patient_ids = patient_ids[: args.max_patients]

    total_slices = 0
    print(f"Starting streaming preprocessing for {len(patient_ids)} patients...")
    for idx, pid in enumerate(patient_ids, 1):
        if preprocessor.is_patient_processed(pid):
            continue
        print(f"[{idx}/{len(patient_ids)}] Downloading & processing {pid}...")
        num_patches = preprocessor.process_patient(pid)
        total_slices += num_patches

    shutil.rmtree(temp_p, ignore_errors=True)
    print(f"\nAll done! Extracted {total_slices} nodule slices directly to {out_p}")


if __name__ == "__main__":
    main()

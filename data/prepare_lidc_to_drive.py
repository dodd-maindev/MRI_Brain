"""End-to-end pipeline downloading, processing, and persisting LIDC to Drive."""

import mus_net.data.numpy_compatibility  # noqa: F401
import argparse
from pathlib import Path
import shutil
from mus_net.data.dataset_manifest import DatasetManifestManager
from mus_net.data.stream_preprocessor import StreamingLidcPreprocessor


def prepare_dataset_pipeline(
    drive_output_dir: str,
    temp_dir: str = "/content/temp_dicom",
    patch_size: int = 128,
    max_patients: int = None,
    start_index: int = 0,
) -> None:
    """Download, process 1st annotation 2D patches, save to Drive, and generate manifest."""
    out_path = Path(drive_output_dir)
    temp_path = Path(temp_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    manifest_mgr = DatasetManifestManager(out_path)
    preprocessor = StreamingLidcPreprocessor(out_path, temp_path, patch_size)

    patient_ids = preprocessor.downloader.fetch_all_patient_ids()
    if start_index > 0:
        patient_ids = patient_ids[start_index:]
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    print(f"=== Starting LIDC-IDRI Pipeline -> Google Drive ===")
    print(f"Target Drive Directory : {out_path}")
    print(f"Total Patients to Check: {len(patient_ids)}")

    processed_new, skipped = 0, 0
    for idx, pid in enumerate(patient_ids, 1):
        if preprocessor.is_patient_processed(pid):
            skipped += 1
            if skipped % 50 == 0 or skipped == 1:
                print(f"[{idx}/{len(patient_ids)}] {pid} already processed. Skipping.")
            continue

        print(f"[{idx}/{len(patient_ids)}] Processing patient: {pid}...")
        extracted_slices = preprocessor.process_patient(pid)
        if extracted_slices > 0:
            processed_new += 1
            print(f"  -> Saved {extracted_slices} nodule slices to Drive.")

    shutil.rmtree(temp_path, ignore_errors=True)
    manifest = manifest_mgr.generate_manifest()
    print(f"\nCompleted! Skipped {skipped} existing patients, processed {processed_new} new patients.")
    print(f"Total Slices on Drive: {manifest.get('total_slices', 0)}")


def main() -> None:
    """Parse CLI arguments for preparing dataset to Drive."""
    parser = argparse.ArgumentParser(description="Prepare LIDC dataset and store on Drive.")
    parser.add_argument("--drive_output_dir", type=str, default="/content/drive/MyDrive/lidc_train_ready")
    parser.add_argument("--temp_dir", type=str, default="/content/temp_dicom")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--max_patients", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0, help="0-based index to start checking from")
    args = parser.parse_args()

    prepare_dataset_pipeline(
        drive_output_dir=args.drive_output_dir,
        temp_dir=args.temp_dir,
        patch_size=args.patch_size,
        max_patients=args.max_patients,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()

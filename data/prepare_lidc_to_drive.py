"""End-to-end pipeline downloading, processing, and persisting LIDC to Drive."""

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
) -> None:
    """Download, process 1st annotation 2D patches, save to Drive, and generate manifest."""
    out_path = Path(drive_output_dir)
    temp_path = Path(temp_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    temp_path.mkdir(parents=True, exist_ok=True)

    manifest_mgr = DatasetManifestManager(out_path)
    preprocessor = StreamingLidcPreprocessor(out_path, temp_path, patch_size)

    patient_ids = preprocessor.downloader.fetch_all_patient_ids()
    if max_patients is not None:
        patient_ids = patient_ids[:max_patients]

    print(f"=== Starting LIDC-IDRI Pipeline -> Google Drive ===")
    print(f"Target Drive Directory : {out_path}")
    print(f"Total Patients to Check: {len(patient_ids)}")

    processed_new, skipped = 0, 0
    for idx, pid in enumerate(patient_ids, 1):
        if preprocessor.is_patient_processed(pid):
            skipped += 1
            continue

        print(f"[{idx}/{len(patient_ids)}] Processing patient: {pid}...")
        extracted_slices = preprocessor.process_patient(pid)
        if extracted_slices > 0:
            processed_new += 1
            print(f"  -> Saved {extracted_slices} nodule slices to Drive.")

    shutil.rmtree(temp_path, ignore_errors=True)
    manifest = manifest_mgr.generate_manifest()

    print(f"\n=======================================================")
    print(f" SUCCESS: Dataset permanently stored on Google Drive!")
    print(f" Location       : {out_path}")
    print(f" Total Slices   : {manifest.get('total_slices', 0)}")
    print(f" Total Patients : {manifest.get('total_patients', 0)}")
    print(f" Format         : int16 HU patch [128, 128], uint8 mask [128, 128]")
    print(f" Manifest       : {out_path / 'dataset_manifest.json'}")
    print(f"=======================================================\n")


def main() -> None:
    """Parse CLI arguments for preparing dataset to Drive."""
    parser = argparse.ArgumentParser(description="Prepare LIDC dataset and store on Drive.")
    parser.add_argument("--drive_output_dir", type=str, default="/content/drive/MyDrive/lidc_train_ready")
    parser.add_argument("--temp_dir", type=str, default="/content/temp_dicom")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--max_patients", type=int, default=None)
    args = parser.parse_args()

    prepare_dataset_pipeline(
        drive_output_dir=args.drive_output_dir,
        temp_dir=args.temp_dir,
        patch_size=args.patch_size,
        max_patients=args.max_patients,
    )


if __name__ == "__main__":
    main()

"""Dataset initialization and DataLoader factory."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from mus_net.data.lidc_dataset import LidcIdriSegmentationDataset
from mus_net.data.patient_splitter import PatientWiseCrossValidationSplitter
from mus_net.data.windowing import ComputedTomographyWindowTransformer


def create_synthetic_lidc_samples(num_samples: int = 50, size: int = 128) -> List[Dict[str, Any]]:
    """Generate deterministic synthetic HU CT patches and nodule masks for testing."""
    samples = []
    np.random.seed(42)
    for i in range(num_samples):
        patient_id = f"LIDC-IDRI-{i // 5:04d}"
        hu_patch = np.random.uniform(-1000.0, 100.0, size=(size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.float32)
        cy, cx = np.random.randint(40, size - 40, size=2)
        radius = np.random.randint(8, 20)
        y, x = np.ogrid[:size, :size]
        nodule_region = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        hu_patch[nodule_region] = np.random.uniform(20.0, 150.0, size=nodule_region.sum())
        mask[nodule_region] = 1.0
        samples.append({"hu_patch": hu_patch, "mask": mask, "patient_id": patient_id})
    return samples


def prepare_fold_dataloaders(
    config: Dict[str, Any],
    fold_index: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for a specific cross-validation fold."""
    data_dir = Path(config["dataset"]["data_directory"])
    samples: List[Dict[str, Any]] = []

    if data_dir.exists() and list(data_dir.glob("*.npz")):
        for file_path in sorted(data_dir.glob("*.npz")):
            data = np.load(file_path)
            samples.append({
                "hu_patch": data["hu_patch"],
                "mask": data["mask"],
                "patient_id": str(data.get("patient_id", file_path.stem)),
            })
    else:
        # Fallback to reproducible synthetic samples if raw LIDC is not yet extracted
        samples = create_synthetic_lidc_samples(num_samples=40, size=config["dataset"]["patch_size"])

    transformer = ComputedTomographyWindowTransformer(
        raw_min=config["dataset"]["raw_min"],
        raw_max=config["dataset"]["raw_max"],
        lung_window_level=config["dataset"]["lung_window_level"],
        lung_window_width=config["dataset"]["lung_window_width"],
        mediastinal_window_level=config["dataset"]["mediastinal_window_level"],
        mediastinal_window_width=config["dataset"]["mediastinal_window_width"],
    )

    splitter = PatientWiseCrossValidationSplitter(
        number_of_folds=config["dataset"]["num_folds"],
        random_seed=config["dataset"]["seed"],
    )
    patient_ids = [s["patient_id"] for s in samples]
    folds = splitter.split(patient_ids)

    train_indices = folds[fold_index]["train_indices"]
    val_indices = folds[fold_index]["validation_indices"]

    train_data = [samples[i] for i in train_indices]
    val_data = [samples[i] for i in val_indices]

    p_size = config["dataset"].get("patch_size", 64)
    train_dataset = LidcIdriSegmentationDataset(train_data, transformer, is_training=True, target_patch_size=p_size)
    val_dataset = LidcIdriSegmentationDataset(val_data, transformer, is_training=False, target_patch_size=p_size)

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


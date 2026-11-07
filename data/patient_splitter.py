"""Patient-wise cross validation splitting to avoid data leakage."""

from typing import Dict, List, Sequence
import numpy as np
from sklearn.model_selection import KFold


class PatientWiseCrossValidationSplitter:
    """Splits samples into K folds grouped by patient identifiers."""

    def __init__(self, number_of_folds: int = 5, random_seed: int = 42) -> None:
        """Initialize K-fold splitter parameters."""
        self.number_of_folds = number_of_folds
        self.random_seed = random_seed

    def split(
        self,
        patient_identifiers: Sequence[str],
    ) -> List[Dict[str, List[int]]]:
        """Generate train and validation index lists for each fold."""
        unique_patients = np.array(sorted(list(set(patient_identifiers))))
        k_fold = KFold(
            n_splits=self.number_of_folds,
            shuffle=True,
            random_state=self.random_seed,
        )

        patient_to_indices: Dict[str, List[int]] = {}
        for index, patient_id in enumerate(patient_identifiers):
            patient_to_indices.setdefault(patient_id, []).append(index)

        fold_splits: List[Dict[str, List[int]]] = []
        for train_patient_indices, val_patient_indices in k_fold.split(
            unique_patients
        ):
            train_indices: List[int] = []
            val_indices: List[int] = []

            for patient_idx in train_patient_indices:
                patient_name = str(unique_patients[patient_idx])
                train_indices.extend(patient_to_indices[patient_name])

            for patient_idx in val_patient_indices:
                patient_name = str(unique_patients[patient_idx])
                val_indices.extend(patient_to_indices[patient_name])

            fold_splits.append({
                "train_indices": train_indices,
                "validation_indices": val_indices,
            })

        return fold_splits


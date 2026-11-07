"""CT Windowing transformations for Hounsfield Unit (HU) normalization."""

from typing import Dict, Union
import numpy as np
import torch


class ComputedTomographyWindowTransformer:
    """Transforms raw CT HU patches into specialized multi-window inputs."""

    def __init__(
        self,
        raw_min: float = -1000.0,
        raw_max: float = 400.0,
        lung_window_level: float = -600.0,
        lung_window_width: float = 1500.0,
        mediastinal_window_level: float = 40.0,
        mediastinal_window_width: float = 400.0,
    ) -> None:
        """Initialize window boundaries and levels.

        ASSUMPTION: Default window level and width values match standard
        radiological lung/mediastinal settings and paper specifications.
        """
        self.raw_min = raw_min
        self.raw_max = raw_max
        self.lung_window_level = lung_window_level
        self.lung_window_width = lung_window_width
        self.mediastinal_window_level = mediastinal_window_level
        self.mediastinal_window_width = mediastinal_window_width

    @staticmethod
    def apply_window(
        image: Union[np.ndarray, torch.Tensor],
        window_level: float,
        window_width: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply linear windowing with clipping to [0, 1]."""
        lower_bound = window_level - window_width / 2.0
        upper_bound = window_level + window_width / 2.0
        if isinstance(image, torch.Tensor):
            scaled = (image - lower_bound) / (upper_bound - lower_bound)
            return torch.clamp(scaled, 0.0, 1.0)
        scaled_numpy = (image - lower_bound) / (upper_bound - lower_bound)
        return np.clip(scaled_numpy, 0.0, 1.0)

    @staticmethod
    def apply_min_max_clip(
        image: Union[np.ndarray, torch.Tensor],
        minimum_value: float,
        maximum_value: float,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Clip image to [minimum_value, maximum_value] and normalize to [0, 1]."""
        if isinstance(image, torch.Tensor):
            clipped = torch.clamp(image, minimum_value, maximum_value)
            return (clipped - minimum_value) / (maximum_value - minimum_value)
        clipped_numpy = np.clip(image, minimum_value, maximum_value)
        return (clipped_numpy - minimum_value) / (maximum_value - minimum_value)

    def transform(
        self,
        hu_image: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Generate three windowed representations from raw HU image."""
        raw_window = self.apply_min_max_clip(
            hu_image, self.raw_min, self.raw_max
        )
        lung_window = self.apply_window(
            hu_image, self.lung_window_level, self.lung_window_width
        )
        mediastinal_window = self.apply_window(
            hu_image,
            self.mediastinal_window_level,
            self.mediastinal_window_width,
        )
        return {
            "raw": raw_window,
            "lung": lung_window,
            "medias": mediastinal_window,
        }


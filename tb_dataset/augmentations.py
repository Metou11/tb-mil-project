"""Albumentations-based augmentation utilities for MIL image datasets.

This module provides an object-oriented augmenter that can be injected into the
TBDataset pipeline. It applies transformations only during training and only
for the configured minority class, which is appropriate for class-imbalanced
chest X-ray datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Albumentations is required. Install it with 'pip install albumentations'."
    ) from exc


ImagePath = Union[str, Path]
Label = Union[str, int]


class DatasetAugmenter:
    """Apply minority-class augmentations to radiography images.

    Parameters
    ----------
    train:
        Whether augmentation is enabled. In validation/test mode, the augmenter
        returns the input image unchanged.
    minority_class:
        Label value or name of the underrepresented class.
    transform:
        Optional Albumentations transform to use. If None, a radiography-safe
        default transform is built automatically.
    """

    def __init__(
        self,
        train: bool = True,
        minority_class: Optional[Label] = "TB",
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.train = train
        self.minority_class = minority_class
        self.transform = transform or self._build_default_transform()

    def _build_default_transform(self) -> A.Compose:
        """Create a radiography-safe augmentation pipeline with Albumentations."""
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.35),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), p=0.25),
            ],
            p=1.0,
        )

    def is_minority(self, label: Label) -> bool:
        """Return True when the label matches the configured minority class."""
        if self.minority_class is None:
            return False
        return str(label).upper() == str(self.minority_class).upper()

    def apply(self, image: np.ndarray, label: Label) -> np.ndarray:
        """Apply augmentation only when training is enabled and the label is minority."""
        if not self.train or self.transform is None:
            return np.asarray(image)
        if not self.is_minority(label):
            return np.asarray(image)

        transformed = self.transform(image=np.asarray(image))
        if isinstance(transformed, dict):
            return np.asarray(transformed["image"])
        return np.asarray(transformed)


__all__ = ["DatasetAugmenter"]

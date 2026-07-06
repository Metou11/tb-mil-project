"""Dataset balancing utilities for chest X-ray tuberculosis detection.

This module provides a reusable augmentation pipeline for binary medical image
classification datasets. It uses Albumentations to augment the minority class
only so that both classes end up with the same number of samples.

The implementation is designed to be compatible with PyTorch/MIL-style pipelines
and can be used alongside preprocessing, patch extraction, and dataset modules.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Albumentations is required. Install it with 'pip install albumentations'."
    ) from exc


ImagePath = Union[str, Path]
Label = str


class DatasetBalancer:
    """Balance a binary dataset by augmenting the minority class only.

    The balancer scans a folder structure like:

    dataset/
        Normal/
        TB/

    and creates a new balanced dataset in an output folder while preserving all
    original images.
    """

    def __init__(
        self,
        input_dir: ImagePath,
        output_dir: ImagePath,
        class_names: Optional[Sequence[str]] = None,
        image_exts: Optional[Sequence[str]] = None,
        seed: int = 42,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_names = list(class_names or ["Normal", "TB"])
        self.image_exts = tuple(image_exts or [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        """Build a radiography-safe augmentation pipeline with Albumentations."""
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.35),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
                A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, p=0.25),
            ],
            p=1.0,
        )

    def _validate_inputs(self) -> None:
        """Check that the input dataset folder and required class folders exist."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")

        for class_name in self.class_names:
            class_dir = self.input_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            if not class_dir.is_dir():
                raise NotADirectoryError(f"Class path is not a directory: {class_dir}")

    def _list_images(self, class_dir: Path) -> List[Path]:
        """List valid image files in a class folder."""
        images = [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.image_exts
        ]
        return sorted(images)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load an image and raise a clear error for corrupted files."""
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Corrupted or unreadable image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save an image to disk without altering the original files."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), image_bgr)

    def _collect_dataset(self) -> Dict[str, List[Path]]:
        """Collect images for each class and validate that the dataset is non-empty."""
        self._validate_inputs()

        dataset: Dict[str, List[Path]] = {}
        for class_name in self.class_names:
            images = self._list_images(self.input_dir / class_name)
            if not images:
                raise ValueError(f"Dataset is empty for class: {class_name}")
            dataset[class_name] = images

        return dataset

    def count_images(self, dataset: Optional[Dict[str, List[Path]]] = None) -> None:
        """Print a summary of the dataset before and after balancing."""
        if dataset is None:
            dataset = self._collect_dataset()

        counts = {name: len(paths) for name, paths in dataset.items()}
        majority_class, minority_class = self._get_class_balance(counts)
        images_to_generate = counts[majority_class] - counts[minority_class]

        print("-" * 34)
        print("Dataset statistics")
        print("-" * 34)
        for class_name in self.class_names:
            print(f"{class_name} : {counts[class_name]}")
        print()
        print(f"Minority class : {minority_class}")
        print(f"Images to generate : {images_to_generate}")
        print()

    def _get_class_balance(self, counts: Dict[str, int]) -> Tuple[str, str]:
        """Identify the majority and minority classes from counts."""
        if len(counts) != 2:
            raise ValueError("Expected a binary dataset with exactly two classes.")

        sorted_counts = sorted(counts.items(), key=lambda item: item[1])
        minority_class, majority_class = sorted_counts[0][0], sorted_counts[1][0]
        return majority_class, minority_class

    def balance_dataset(self) -> Dict[str, List[Path]]:
        """Balance the dataset by augmenting the minority class only.

        The method creates a new output directory structure:

        balanced_dataset/
            Normal/
            TB/

        and preserves all original images in the input dataset.
        """
        dataset = self._collect_dataset()
        counts = {name: len(paths) for name, paths in dataset.items()}
        majority_class, minority_class = self._get_class_balance(counts)
        images_to_generate = counts[majority_class] - counts[minority_class]

        if images_to_generate <= 0:
            raise ValueError("Dataset is already balanced or the minority class is not smaller.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for class_name in self.class_names:
            class_output_dir = self.output_dir / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy original images to the output dataset.
        for class_name in self.class_names:
            source_dir = self.input_dir / class_name
            target_dir = self.output_dir / class_name
            for image_path in self._list_images(source_dir):
                target_path = target_dir / image_path.name
                self._save_image(self._load_image(image_path), target_path)

        # Generate additional images for the minority class only.
        minority_images = dataset[minority_class]
        for index in range(images_to_generate):
            source_image = minority_images[index % len(minority_images)]
            image = self._load_image(source_image)
            augmented = self.transform(image=image)["image"]
            output_name = f"{minority_class}_{index + 1:04d}_aug.png"
            self._save_image(augmented, self.output_dir / minority_class / output_name)

        self.count_images({name: self._list_images(self.output_dir / name) for name in self.class_names})
        return {name: self._list_images(self.output_dir / name) for name in self.class_names}


__all__ = ["DatasetBalancer"]

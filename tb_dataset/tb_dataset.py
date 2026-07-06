"""PyTorch MIL dataset orchestration for tuberculosis detection.

This module is intentionally thin: it uses dependency injection to delegate
preprocessing, augmentation and patch extraction to dedicated classes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .patch_extraction import PatchExtractor
from .preprocessing import Preprocessor
from .augmentations import DatasetAugmenter

ImagePath = Union[str, Path]
LabelValue = Union[int, float, str]


class TBDataset(Dataset):
    """Create MIL bags from chest X-ray images stored in a CSV file.

    Each sample is processed through the injected preprocessor, optional
    minority-class augmenter and patch extractor. The dataset returns a bag of
    patches together with the label, coordinates and metadata required by MIL
    models.
    """

    def __init__(
        self,
        csv_file: ImagePath,
        preprocessor: Preprocessor,
        patch_extractor: PatchExtractor,
        augmenter: Optional[DatasetAugmenter] = None,
        train: bool = True,
        image_column: str = "image_path",
        label_column: str = "label",
        patch_size: Tuple[int, int] = (256, 256),
        overlap: float = 0.0,
        ignore_empty: bool = False,
        empty_threshold: float = 0.05,
        minority_class: str = "TB",
        label_mapping: Optional[Dict[str, int]] = None,
        cache: bool = False,
    ) -> None:
        """Initialize the dataset from a CSV file and injected processing objects."""
        self.csv_file = Path(csv_file)
        self.preprocessor = preprocessor
        self.patch_extractor = patch_extractor
        self.augmenter = augmenter
        self.train = train
        self.image_column = image_column
        self.label_column = label_column
        self.patch_size = patch_size
        self.overlap = overlap
        self.ignore_empty = ignore_empty
        self.empty_threshold = empty_threshold
        self.minority_class = minority_class
        self.label_mapping = label_mapping or {"TB": 1, "Normal": 0}
        self.cache = cache

        self.samples = self._load_csv_samples()
        self._cache: Dict[int, Dict[str, Any]] = {}

    def _load_csv_samples(self) -> List[Dict[str, Any]]:
        """Validate and load samples from the CSV file."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        with self.csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("The CSV file is empty or has no header row.")

            required_columns = {self.image_column, self.label_column}
            missing_columns = required_columns.difference(set(reader.fieldnames))
            if missing_columns:
                raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

            rows: List[Dict[str, Any]] = []
            for row_index, row in enumerate(reader, start=2):
                image_path_value = (row.get(self.image_column) or "").strip()
                if not image_path_value:
                    raise ValueError(f"Empty image path at row {row_index}.")

                image_path = Path(image_path_value)
                if not image_path.is_absolute():
                    image_path = (self.csv_file.parent / image_path).resolve()
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                label_value = self._parse_label(row.get(self.label_column))
                rows.append({"image_path": str(image_path), "label": label_value})

            if not rows:
                raise ValueError("The CSV file contains no valid samples.")

            return rows

    def _parse_label(self, label_raw: Optional[LabelValue]) -> int:
        """Convert a label from the CSV file to the expected integer format."""
        if label_raw is None:
            raise ValueError("Label value is missing.")

        if isinstance(label_raw, (int, np.integer)):
            label_int = int(label_raw)
            if label_int not in {0, 1}:
                raise ValueError(f"Invalid label {label_int}; expected 0 or 1.")
            return label_int

        if isinstance(label_raw, (float, np.floating)):
            label_int = int(label_raw)
            if label_int not in {0, 1}:
                raise ValueError(f"Invalid label {label_int}; expected 0 or 1.")
            return label_int

        text_label = str(label_raw).strip()
        if text_label in self.label_mapping:
            return int(self.label_mapping[text_label])

        normalized = text_label.upper()
        if normalized == "TB":
            return 1
        if normalized == "NORMAL":
            return 0
        raise ValueError(f"Invalid label {label_raw}; expected one of {sorted(self.label_mapping)} or 0/1.")

    def _apply_augmentation(self, image: np.ndarray, label: int) -> np.ndarray:
        """Apply augmentation only for training and for the minority class."""
        if self.augmenter is None or not self.train:
            return image
        return self.augmenter.apply(image, label)

    def _prepare_sample(self, index: int) -> Dict[str, Any]:
        """Run preprocessing, augmentation and patch extraction for one sample."""
        sample = self.samples[index]
        image_path = Path(sample["image_path"])

        try:
            image_array = self.preprocessor.preprocess(image_path)
        except Exception as exc:  # pragma: no cover - runtime validation path
            raise ValueError(f"Unable to preprocess image at {image_path}: {exc}") from exc

        if image_array.ndim != 3:
            raise ValueError(f"Expected an RGB image after preprocessing, got shape {image_array.shape}.")

        try:
            image_array = self._apply_augmentation(image_array, int(sample["label"]))
        except Exception as exc:  # pragma: no cover - runtime validation path
            raise ValueError(f"Unable to augment image at {image_path}: {exc}") from exc

        try:
            patches, coords = self.patch_extractor.extract(
                image_array,
                patch_size=self.patch_size,
                overlap=self.overlap,
                ignore_empty=self.ignore_empty,
                empty_threshold=self.empty_threshold,
            )
        except Exception as exc:  # pragma: no cover - runtime validation path
            raise ValueError(f"Unable to extract patches for {image_path}: {exc}") from exc

        if not patches:
            raise ValueError(f"Bag is empty after patch extraction for {image_path}.")

        patch_tensors = [torch.from_numpy(np.transpose(np.asarray(patch), (2, 0, 1))).float() for patch in patches]
        patches_tensor = torch.stack(patch_tensors, dim=0)
        label_tensor = torch.tensor(int(sample["label"]), dtype=torch.long)

        return {
            "patches": patches_tensor,
            "coords": coords,
            "label": label_tensor,
            "path": str(image_path),
            "num_patches": int(patches_tensor.size(0)),
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return one sample as a MIL bag-compatible dictionary."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")

        if self.cache and index in self._cache:
            return self._cache[index]

        prepared = self._prepare_sample(index)
        if self.cache:
            self._cache[index] = prepared
        return prepared

    def get_class_distribution(self) -> Dict[str, int]:
        """Return the class distribution present in the CSV file."""
        counts: Dict[str, int] = {}
        for sample in self.samples:
            label = int(sample["label"])
            class_name = "TB" if label == 1 else "Normal"
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def print_dataset_summary(self) -> None:
        """Print a concise summary of the dataset content."""
        distribution = self.get_class_distribution()
        print("-" * 34)
        print("Dataset summary")
        print("-" * 34)
        for class_name in ["Normal", "TB"]:
            print(f"{class_name} : {distribution.get(class_name, 0)}")
        print()
        print(f"Samples : {len(self)}")
        print(f"Mode : {'train' if self.train else 'eval'}")
        print()


def collate_mil_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of variable-sized MIL bags without padding."""
    labels = torch.stack([item["label"] for item in batch], dim=0)
    return {
        "patches": [item["patches"] for item in batch],
        "coords": [item["coords"] for item in batch],
        "label": labels,
        "path": [item["path"] for item in batch],
        "num_patches": [item["num_patches"] for item in batch],
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    preprocessor = Preprocessor(target_size=(512, 512), normalization="imagenet")
    patch_extractor = PatchExtractor(patch_size=(256, 256), overlap=0.0)
    augmenter = DatasetAugmenter(train=True, minority_class="TB")

    dataset = TBDataset(
        csv_file="data/metadata.csv",
        preprocessor=preprocessor,
        patch_extractor=patch_extractor,
        augmenter=augmenter,
    )
    dataset.print_dataset_summary()

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_mil_batch)
    batch = next(iter(loader))
    print(batch["label"].shape)
    print(batch["num_patches"])


__all__ = ["TBDataset", "collate_mil_batch"]

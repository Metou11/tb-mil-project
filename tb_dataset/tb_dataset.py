"""PyTorch dataset orchestration for MIL-based tuberculosis detection.

This module acts as the integration layer between the preprocessing,
augmentation and patch extraction components. It does not reimplement their
logic; instead, it injects those dependencies and orchestrates them in a
PyTorch Dataset compatible with MIL pipelines such as ABMIL, CLAM, DSMIL and
TransMIL.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

try:
    from preprocessing import preprocess_image  # type: ignore
except ImportError:  # pragma: no cover - optional import
    preprocess_image = None

try:
    from patch_extraction import extract_patches_from_image  # type: ignore
except ImportError:  # pragma: no cover - optional import
    extract_patches_from_image = None


ImagePath = Union[str, Path]
LabelValue = Union[int, float, str]


class TBDataset(Dataset):
    """PyTorch Dataset for MIL bags built from chest X-ray images.

    The dataset reads a CSV file containing image paths and labels, applies the
    configured preprocessing and optional minority-class augmentation, extracts
    patches, and returns a bag of patches for each image. The dataset is
    designed to be compatible with MIL models that operate on bags of patches.

    The implementation is intentionally modular: preprocessing, augmentation and
    patch extraction are injected from outside and are not reimplemented here.
    """

    def __init__(
        self,
        csv_file: ImagePath,
        preprocessor: Optional[Callable[[ImagePath], np.ndarray]] = None,
        patch_extractor: Optional[Callable[..., Any]] = None,
        augmenter: Optional[Any] = None,
        train: bool = True,
        image_column: str = "image_path",
        label_column: str = "label",
        patch_size: Tuple[int, int] = (256, 256),
        overlap: float = 0.0,
        ignore_empty: bool = False,
        empty_threshold: float = 0.05,
        minority_class: str = "TB",
        label_mapping: Optional[Dict[str, int]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize the dataset from a CSV file and injected processing modules."""
        self.csv_file = Path(csv_file)
        self.preprocessor = self._resolve_preprocessor(preprocessor)
        self.patch_extractor = self._resolve_patch_extractor(patch_extractor)
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
        self.class_distribution = self.get_class_distribution()

    def _resolve_preprocessor(self, preprocessor: Optional[Callable[[ImagePath], np.ndarray]]) -> Callable[[ImagePath], np.ndarray]:
        """Resolve a preprocessor callable from an injected object or a fallback helper."""
        if preprocessor is not None:
            if callable(preprocessor):
                return preprocessor
            if hasattr(preprocessor, "preprocess") and callable(preprocessor.preprocess):
                return preprocessor.preprocess
            if hasattr(preprocessor, "preprocess_image") and callable(preprocessor.preprocess_image):
                return preprocessor.preprocess_image

        if preprocess_image is not None:
            return preprocess_image

        raise ValueError("A preprocessor must be provided or available as preprocess_image().")

    def _resolve_patch_extractor(self, patch_extractor: Optional[Callable[..., Any]]) -> Callable[..., Any]:
        """Resolve a patch extractor callable from an injected object or a fallback helper."""
        if patch_extractor is not None:
            if callable(patch_extractor):
                return patch_extractor
            if hasattr(patch_extractor, "extract_patches_from_image") and callable(patch_extractor.extract_patches_from_image):
                return patch_extractor.extract_patches_from_image
            if hasattr(patch_extractor, "extract_patches") and callable(patch_extractor.extract_patches):
                return patch_extractor.extract_patches

        if extract_patches_from_image is not None:
            return extract_patches_from_image

        raise ValueError("A patch extractor must be provided or available as extract_patches_from_image().")

    def _load_csv_samples(self) -> List[Dict[str, Any]]:
        """Load the CSV file and validate its structure."""
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
                image_path = (row.get(self.image_column) or "").strip()
                if not image_path:
                    raise ValueError(f"Empty image path at row {row_index}.")

                image_path = (self.csv_file.parent / image_path).resolve() if not Path(image_path).is_absolute() else Path(image_path)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file not found: {image_path}")

                label_raw = row.get(self.label_column)
                label_value = self._parse_label(label_raw)
                rows.append({"image_path": str(image_path), "label": label_value})

            if not rows:
                raise ValueError("The CSV file contains no valid samples.")

            return rows

    def _parse_label(self, label_raw: Optional[LabelValue]) -> int:
        """Convert a label from the CSV to an integer value expected by PyTorch."""
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

    def _is_minority_label(self, label: int) -> bool:
        """Check whether the current label matches the configured minority class."""
        if not self.label_mapping:
            return False
        for name, value in self.label_mapping.items():
            if value == label and name == self.minority_class:
                return True
        return False

    def _apply_augmentation(self, image: np.ndarray, label: int) -> np.ndarray:
        """Apply augmentation only in training mode and for the minority class."""
        if not self.train or self.augmenter is None:
            return image
        if not self._is_minority_label(label):
            return image

        if hasattr(self.augmenter, "transform") and getattr(self.augmenter, "transform", None) is not None:
            transformed = self.augmenter.transform(image=image)
            if isinstance(transformed, dict):
                return transformed["image"]
            return transformed

        if hasattr(self.augmenter, "augment_sample") and callable(self.augmenter.augment_sample):
            augmented_image, _ = self.augmenter.augment_sample(image, label)
            return np.asarray(augmented_image)

        if callable(self.augmenter):
            return np.asarray(self.augmenter(image))

        return image

    def _prepare_sample(self, index: int) -> Dict[str, Any]:
        """Prepare one sample by running preprocessing, augmentation and patch extraction."""
        sample = self.samples[index]
        image_path = Path(sample["image_path"])

        try:
            preprocessed = self.preprocessor(image_path)
            image_array = np.asarray(preprocessed)
        except Exception as exc:  # pragma: no cover - runtime validation path
            raise ValueError(f"Unable to preprocess image at {image_path}: {exc}") from exc

        if image_array.ndim != 3:
            raise ValueError(f"Expected an RGB image after preprocessing, got shape {image_array.shape}.")

        try:
            image_array = self._apply_augmentation(image_array, int(sample["label"]))
        except Exception as exc:  # pragma: no cover - runtime validation path
            raise ValueError(f"Unable to augment image at {image_path}: {exc}") from exc

        try:
            patches, coords = self.patch_extractor(
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
        if not patch_tensors:
            raise ValueError(f"No patches produced for {image_path}.")

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
        """Return one MIL bag as a dictionary suitable for PyTorch training."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {len(self)}.")

        if self.cache and index in self._cache:
            return self._cache[index]

        prepared = self._prepare_sample(index)
        if self.cache:
            self._cache[index] = prepared
        return prepared

    def get_class_distribution(self) -> Dict[str, int]:
        """Return a summary of the class distribution in the loaded CSV file."""
        counts: Dict[str, int] = {}
        for sample in self.samples:
            label = int(sample["label"])
            class_name = "TB" if label == 1 else "Normal"
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def print_dataset_summary(self) -> None:
        """Print a compact summary of the loaded dataset."""
        distribution = self.get_class_distribution()
        print("-" * 34)
        print("Dataset summary")
        print("-" * 34)
        for class_name in ["Normal", "TB"]:
            count = distribution.get(class_name, 0)
            print(f"{class_name} : {count}")
        print()
        print(f"Samples : {len(self)}")
        print(f"Mode : {'train' if self.train else 'eval'}")
        print()


def collate_mil_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple collate function for variable-length MIL bags.

    This function preserves the per-sample patch lists instead of padding them,
    which is appropriate for MIL pipelines that handle bags of variable size.
    """
    labels = torch.stack([item["label"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    coords = [item["coords"] for item in batch]
    num_patches = [item["num_patches"] for item in batch]
    patches = [item["patches"] for item in batch]
    return {"patches": patches, "coords": coords, "label": labels, "path": paths, "num_patches": num_patches}


if __name__ == "__main__":
    """Minimal end-to-end example for using the dataset with a preprocessor, patch extractor and augmenter."""
    from torch.utils.data import DataLoader

    # Example placeholders; replace them with your own project components.
    preprocessor = preprocess_image
    patch_extractor = extract_patches_from_image

    try:
        from augmentation import DatasetBalancer

        augmenter = DatasetBalancer(input_dir="dataset", output_dir="balanced_dataset")
    except Exception:  # pragma: no cover - optional dependency path
        augmenter = None

    dataset = TBDataset(
        csv_file="data/metadata.csv",
        preprocessor=preprocessor,
        patch_extractor=patch_extractor,
        augmenter=augmenter,
        train=True,
        patch_size=(256, 256),
        overlap=0.0,
    )
    dataset.print_dataset_summary()

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_mil_batch)
    batch = next(iter(loader))
    print(type(batch["patches"]))
    print(batch["label"].shape)
    print(batch["num_patches"])


__all__ = ["TBDataset", "collate_mil_batch"]

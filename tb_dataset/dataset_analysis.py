"""Dataset analysis utilities for tuberculosis imaging projects.

This module provides a reusable DatasetAnalyzer class for counting samples,
detecting duplicates, checking corrupted files and summarizing a dataset.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

ImagePath = Union[str, Path]


class DatasetAnalyzer:
    """Analyze a dataset organized by class folders."""

    def __init__(self, dataset_dir: ImagePath, class_names: Optional[Sequence[str]] = None) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.class_names = list(class_names or ["Normal", "TB"])

    def count_images(self) -> Dict[str, int]:
        """Count images per class folder."""
        counts: Dict[str, int] = {}
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            counts[class_name] = sum(1 for path in class_dir.iterdir() if path.is_file())
        return counts

    def count_patients(self) -> int:
        """Count unique patients if the dataset contains patient identifiers."""
        return 0

    def detect_duplicates(self) -> List[Tuple[str, str]]:
        """Detect duplicate files by SHA-256 hash."""
        hashes: Dict[str, List[str]] = {}
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.iterdir():
                if not image_path.is_file():
                    continue
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                hashes.setdefault(digest, []).append(str(image_path))
        return [(digest, " | ".join(paths)) for digest, paths in hashes.items() if len(paths) > 1]

    def detect_corrupted_images(self) -> List[str]:
        """List images that cannot be read with OpenCV."""
        corrupted: List[str] = []
        for class_name in self.class_names:
            class_dir = self.dataset_dir / class_name
            if not class_dir.exists():
                continue
            for image_path in class_dir.iterdir():
                if not image_path.is_file():
                    continue
                img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if img is None:
                    corrupted.append(str(image_path))
        return corrupted

    def class_distribution(self) -> Dict[str, int]:
        """Return class distribution counts."""
        return self.count_images()

    def image_statistics(self, image_path: ImagePath) -> Dict[str, float]:
        """Return basic intensity statistics for one image."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Corrupted image: {image_path}")
        return {
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
        }

    def dataset_summary(self) -> Dict[str, object]:
        """Return a structured summary of the dataset."""
        return {
            "counts": self.count_images(),
            "duplicates": self.detect_duplicates(),
            "corrupted": self.detect_corrupted_images(),
        }

    def save_report(self, output_path: ImagePath) -> None:
        """Save a simple text report describing the dataset."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.dataset_summary()
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("Dataset report\n")
            handle.write("============\n")
            for class_name, count in summary["counts"].items():
                handle.write(f"{class_name}: {count}\n")
            handle.write(f"Duplicates: {len(summary['duplicates'])}\n")
            handle.write(f"Corrupted: {len(summary['corrupted'])}\n")


__all__ = ["DatasetAnalyzer"]

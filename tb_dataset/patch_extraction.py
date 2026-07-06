"""Patch extraction utilities for MIL pipelines.

The module is organized around a single PatchExtractor class that encapsulates
sliding window patch generation and bag creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

PatchSize = Union[int, Tuple[int, int]]


class PatchExtractor:
    """Extract patches from an image and create MIL bags.

    This implementation preserves the original sliding-window logic while making
    it reusable through an object-oriented interface.
    """

    def __init__(
        self,
        patch_size: PatchSize = (256, 256),
        overlap: Union[float, Tuple[float, float]] = 0.0,
        ignore_empty: bool = False,
        empty_threshold: float = 0.05,
    ) -> None:
        """Configure the patch extraction parameters."""
        self.patch_size = patch_size
        self.overlap = overlap
        self.ignore_empty = ignore_empty
        self.empty_threshold = empty_threshold

    def _normalize_patch_size(self, patch_size: Optional[PatchSize] = None) -> Tuple[int, int]:
        """Normalize patch size to a height/width tuple."""
        size = self.patch_size if patch_size is None else patch_size
        if isinstance(size, int):
            return size, size
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return int(size[0]), int(size[1])
        raise ValueError("patch_size doit être un entier ou un tuple (h, w)")

    def _normalize_overlap(self, overlap: Optional[Union[float, Tuple[float, float]]] = None) -> Tuple[float, float]:
        """Normalize overlap to a height/width tuple."""
        overlap_value = self.overlap if overlap is None else overlap
        if isinstance(overlap_value, (tuple, list)):
            if len(overlap_value) != 2:
                raise ValueError("overlap doit être un float ou un tuple (vh, vw)")
            ov_h, ov_w = float(overlap_value[0]), float(overlap_value[1])
        else:
            ov_h = ov_w = float(overlap_value)

        if not 0.0 <= ov_h < 1.0 or not 0.0 <= ov_w < 1.0:
            raise ValueError("overlap doit être compris entre 0.0 et 1.0 (exclu)")
        return ov_h, ov_w

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensure the input image is RGB."""
        image = np.asarray(image)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 1:
            return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError("image doit être en niveaux de gris ou RGB")

    def _load_image(self, image_or_path: Union[np.ndarray, str, Path]) -> np.ndarray:
        """Load an image from disk or accept an array directly."""
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Image introuvable : {image_or_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self._ensure_rgb(image_or_path)

    def _is_empty_patch(self, patch: np.ndarray, empty_threshold: float = 0.05) -> bool:
        """Check whether a patch is mostly empty."""
        if patch.size == 0:
            return True
        gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        non_zero = np.count_nonzero(gray)
        return (non_zero / gray.size) < empty_threshold

    def extract(
        self,
        image: Union[np.ndarray, str, Path],
        patch_size: Optional[PatchSize] = None,
        overlap: Optional[Union[float, Tuple[float, float]]] = None,
        ignore_empty: Optional[bool] = None,
        empty_threshold: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract patches and their positions from an image."""
        image_arr = self._load_image(image)
        patch_h, patch_w = self._normalize_patch_size(patch_size)
        ov_h, ov_w = self._normalize_overlap(overlap)
        use_ignore_empty = self.ignore_empty if ignore_empty is None else ignore_empty
        use_empty_threshold = self.empty_threshold if empty_threshold is None else empty_threshold

        image_arr = self._ensure_rgb(image_arr)
        height, width = image_arr.shape[:2]

        if patch_h > height or patch_w > width:
            pad_h = max(patch_h - height, 0)
            pad_w = max(patch_w - width, 0)
            image_arr = cv2.copyMakeBorder(
                image_arr,
                0,
                pad_h,
                0,
                pad_w,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            height, width = image_arr.shape[:2]

        stride_h = max(1, int(round(patch_h * (1.0 - ov_h))))
        stride_w = max(1, int(round(patch_w * (1.0 - ov_w))))

        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []

        y_positions = list(range(0, height - patch_h + 1, stride_h))
        if not y_positions or y_positions[-1] != height - patch_h:
            y_positions.append(height - patch_h)

        x_positions = list(range(0, width - patch_w + 1, stride_w))
        if not x_positions or x_positions[-1] != width - patch_w:
            x_positions.append(width - patch_w)

        for y0 in y_positions:
            for x0 in x_positions:
                patch = image_arr[y0:y0 + patch_h, x0:x0 + patch_w]
                if use_ignore_empty and self._is_empty_patch(patch, empty_threshold=use_empty_threshold):
                    continue
                patches.append(patch)
                coords.append((y0, x0))

        return patches, coords

    def create_bag(
        self,
        images: Iterable[Union[np.ndarray, str, Path]],
        patch_size: Optional[PatchSize] = None,
        overlap: Optional[Union[float, Tuple[float, float]]] = None,
        ignore_empty: Optional[bool] = None,
        empty_threshold: Optional[float] = None,
    ) -> List[List[np.ndarray]]:
        """Create a MIL bag for each image in an iterable."""
        bags: List[List[np.ndarray]] = []
        for image in images:
            patches, _ = self.extract(
                image,
                patch_size=patch_size,
                overlap=overlap,
                ignore_empty=ignore_empty,
                empty_threshold=empty_threshold,
            )
            bags.append(patches)
        return bags

    def bag_to_tensor(self, bag: Sequence[np.ndarray], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert a bag of patches to a PyTorch tensor."""
        if len(bag) == 0:
            raise ValueError("La bag MIL est vide")
        patch_tensors = [torch.from_numpy(np.transpose(np.asarray(patch), (2, 0, 1))).to(dtype) for patch in bag]
        return torch.stack(patch_tensors, dim=0)


__all__ = ["PatchExtractor"]

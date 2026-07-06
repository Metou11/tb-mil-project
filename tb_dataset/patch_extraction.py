"""
Module de découpage d'images en patches pour pipelines MIL.

Fonctionnalités :
- découpage d'images en patches
- gestion du chevauchement (overlap)
- création de bags MIL
- option pour ignorer les patches vides / majoritairement noirs
- sortie compatible avec des architectures MIL telles que ABMIL, CLAM, TransMIL et DSMIL
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path

import cv2
import numpy as np

PatchSize = Union[int, Tuple[int, int]]


def _normalize_patch_size(patch_size: PatchSize) -> Tuple[int, int]:
    if isinstance(patch_size, int):
        return patch_size, patch_size
    if isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
        return int(patch_size[0]), int(patch_size[1])
    raise ValueError("patch_size doit être un entier ou un tuple (h, w)")


def _normalize_overlap(overlap: Union[float, Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(overlap, (tuple, list)):
        if len(overlap) != 2:
            raise ValueError("overlap doit être un float ou un tuple (vh, vw)")
        ov_h, ov_w = float(overlap[0]), float(overlap[1])
    else:
        ov_h = ov_w = float(overlap)

    if not 0.0 <= ov_h < 1.0 or not 0.0 <= ov_w < 1.0:
        raise ValueError("overlap doit être compris entre 0.0 et 1.0 (exclu)")
    return ov_h, ov_w


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError("image doit être en niveaux de gris ou RGB")


def _load_image(image_or_path: Union[np.ndarray, str, Path]) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        img = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image introuvable : {image_or_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _ensure_rgb(image_or_path)


def _is_empty_patch(patch: np.ndarray, empty_threshold: float = 0.05) -> bool:
    if patch.size == 0:
        return True
    gray = patch if patch.ndim == 2 else cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    non_zero = np.count_nonzero(gray)
    return (non_zero / gray.size) < empty_threshold


def extract_patches_from_image(
    image: Union[np.ndarray, str, Path],
    patch_size: PatchSize = (256, 256),
    overlap: Union[float, Tuple[float, float]] = 0.0,
    ignore_empty: bool = False,
    empty_threshold: float = 0.05,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Découpe une image en patches et retourne les patches + leurs coordonnées.

    Args:
        image: image numpy RGB/gray ou chemin vers une image.
        patch_size: taille du patch (h, w) ou un entier pour un carré.
        overlap: chevauchement entre 0.0 et 1.0 (ou (vh, vw)).
        ignore_empty: si True, ignore les patches majoritairement noirs.
        empty_threshold: seuil de non-nullité pour considérer un patch comme vide.

    Returns:
        (patches, coords)
    """
    image_arr = _load_image(image)
    patch_h, patch_w = _normalize_patch_size(patch_size)
    ov_h, ov_w = _normalize_overlap(overlap)

    image_arr = _ensure_rgb(image_arr)
    h, w = image_arr.shape[:2]

    if patch_h > h or patch_w > w:
        pad_h = max(patch_h - h, 0)
        pad_w = max(patch_w - w, 0)
        image_arr = cv2.copyMakeBorder(
            image_arr,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        h, w = image_arr.shape[:2]

    stride_h = max(1, int(round(patch_h * (1.0 - ov_h))))
    stride_w = max(1, int(round(patch_w * (1.0 - ov_w))))

    patches: List[np.ndarray] = []
    coords: List[Tuple[int, int]] = []

    y_positions = list(range(0, h - patch_h + 1, stride_h))
    if not y_positions or y_positions[-1] != h - patch_h:
        y_positions.append(h - patch_h)

    x_positions = list(range(0, w - patch_w + 1, stride_w))
    if not x_positions or x_positions[-1] != w - patch_w:
        x_positions.append(w - patch_w)

    for y0 in y_positions:
        for x0 in x_positions:
            patch = image_arr[y0:y0 + patch_h, x0:x0 + patch_w]
            if ignore_empty and _is_empty_patch(patch, empty_threshold=empty_threshold):
                continue
            patches.append(patch)
            coords.append((y0, x0))

    return patches, coords


def create_mil_bags(
    images: Iterable[Union[np.ndarray, str, Path]],
    patch_size: PatchSize = (256, 256),
    overlap: Union[float, Tuple[float, float]] = 0.0,
    ignore_empty: bool = False,
    empty_threshold: float = 0.05,
) -> List[List[np.ndarray]]:
    """
    Crée une liste de bags MIL, une bag par image.

    Chaque bag est une liste de patches, ce qui est adapté aux architectures MIL
    telles que ABMIL, CLAM, TransMIL ou DSMIL.
    """
    bags: List[List[np.ndarray]] = []
    for image in images:
        patches, _ = extract_patches_from_image(
            image,
            patch_size=patch_size,
            overlap=overlap,
            ignore_empty=ignore_empty,
            empty_threshold=empty_threshold,
        )
        bags.append(patches)
    return bags


def bag_to_array(bag: Sequence[np.ndarray], dtype: np.dtype = np.float32) -> np.ndarray:
    """Convertit une bag MIL en tableau NumPy de forme (n_patches, h, w, c)."""
    if len(bag) == 0:
        raise ValueError("La bag MIL est vide")
    return np.stack([np.asarray(patch).astype(dtype) for patch in bag], axis=0)


__all__ = [
    "extract_patches_from_image",
    "create_mil_bags",
    "bag_to_array",
]

"""
preprocessing.py
----------------
Pipeline complet de prétraitement pour radiographies pulmonaires TB.

Étapes :
1. Lecture image
2. CLAHE (amélioration contraste)
3. Redimensionnement
4. Normalisation (ImageNet ou [0,1])

Compatible ResNet / ViT / Swin / MIL pipelines.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# =========================================================
# 📊 ImageNet stats
# =========================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================
# 1. LOAD IMAGE
# =========================================================
def load_image(path: str, force_rgb: bool = True) -> np.ndarray:
    """
    Lecture image robuste (grayscale → RGB).
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Image introuvable: {path}")

    if force_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


# =========================================================
# 2. CLAHE (contrast enhancement)
# =========================================================
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # grayscale
    if image.ndim == 2:
        return clahe.apply(image)

    # RGB → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# =========================================================
# 3. RESIZE
# =========================================================
def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:

    h, w = target_size

    interp = cv2.INTER_AREA if image.shape[0] > h else cv2.INTER_CUBIC

    return cv2.resize(image, (w, h), interpolation=interp)


# =========================================================
# 4. NORMALIZATION
# =========================================================
def normalize_image(
    image: np.ndarray,
    method: str = "imagenet"
) -> np.ndarray:

    image = image.astype(np.float32) / 255.0

    if method == "imagenet":
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

    elif method == "unit":
        pass

    else:
        raise ValueError(f"Unknown normalization: {method}")

    return image


# =========================================================
# 5. FULL PIPELINE
# =========================================================
def preprocess_image(
    path: str,
    target_size: Tuple[int, int] = (512, 512),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    normalization: Optional[str] = "imagenet",
) -> np.ndarray:

    # 1. load
    image = load_image(path, force_rgb=True)

    # 2. CLAHE
    image = apply_clahe(
        image,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_grid_size
    )

    # 3. resize
    image = resize_image(image, target_size)

    # 4. normalization
    if normalization is not None:
        image = normalize_image(image, method=normalization)

    return image
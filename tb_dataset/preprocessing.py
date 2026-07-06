"""
preprocessing.py
----------------
Pipeline complet de prétraitement pour radiographies pulmonaires TB.

Étapes :
1. Lecture image
2. CLAHE (amélioration du contraste)
3. Redimensionnement
4. Normalisation (ImageNet ou [0,1])

Compatible ResNet / ViT / Swin / MIL.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class Preprocessor:
    """
    Pipeline de prétraitement des radiographies pulmonaires.

    Exemple
    --------
    >>> preprocessor = Preprocessor(target_size=(512,512))
    >>> image = preprocessor.preprocess("image.png")
    """

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalization: Optional[str] = "imagenet",
        force_rgb: bool = True,
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.normalization = normalization
        self.force_rgb = force_rgb

    # =====================================================
    # 1. Chargement
    # =====================================================
    def load_image(self, path: str) -> np.ndarray:
        """
        Charge une radiographie.

        Parameters
        ----------
        path : str
            Chemin de l'image.

        Returns
        -------
        np.ndarray
        """
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Impossible de charger : {path}")

        if self.force_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image

    # =====================================================
    # 2. CLAHE
    # =====================================================
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Amélioration locale du contraste.
        """

        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

        if image.ndim == 2:
            return clahe.apply(image)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # =====================================================
    # 3. Resize
    # =====================================================
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Redimensionnement.
        """

        h, w = self.target_size

        interpolation = (
            cv2.INTER_AREA
            if image.shape[0] > h
            else cv2.INTER_CUBIC
        )

        return cv2.resize(image, (w, h), interpolation=interpolation)

    # =====================================================
    # 4. Normalisation
    # =====================================================
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalisation de l'image.
        """

        image = image.astype(np.float32) / 255.0

        if self.normalization == "imagenet":

            image = (
                image - self.IMAGENET_MEAN
            ) / self.IMAGENET_STD

        elif self.normalization == "unit":
            pass

        elif self.normalization is None:
            return image

        else:
            raise ValueError(
                f"Unknown normalization : {self.normalization}"
            )

        return image

    # =====================================================
    # Pipeline complet
    # =====================================================
    def preprocess(self, path: str) -> np.ndarray:
        """
        Pipeline complet.

        image
            ↓
        load
            ↓
        CLAHE
            ↓
        resize
            ↓
        normalization
        """

        image = self.load_image(path)

        image = self.apply_clahe(image)

        image = self.resize_image(image)

        if self.normalization is not None:
            image = self.normalize_image(image)

        return image
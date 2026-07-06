"""Backward-compatible augmentation wrapper.

The project now uses the object-oriented module augmentations.py; this file
keeps the older import path working for compatibility.
"""

from .augmentations import DatasetAugmenter

DatasetBalancer = DatasetAugmenter

__all__ = ["DatasetAugmenter", "DatasetBalancer"]

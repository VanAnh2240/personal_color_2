"""
classification/__init__.py  (fixed v2)

Bỏ anti-pattern "PaletteClassifier = None" và try/except ẩn ImportError.
Nếu dependencies chưa cài, lỗi sẽ nổi rõ khi import thay vì im lặng.
"""

from .palettes import ALL_SEASONS, SPRING, SUMMER, AUTUMN, WINTER, SeasonPalette
from .classifier import PaletteClassifier, ClassificationResult
from .color_utils import (
    rgb_to_lab,
    lab_distance,
    extract_dominant_color,
    classify_hue,
    classify_chroma,
    classify_value,
    classify_contrast,
)

__all__ = [
    "PaletteClassifier",
    "ClassificationResult",
    "SeasonPalette",
    "ALL_SEASONS",
    "SPRING",
    "SUMMER",
    "AUTUMN",
    "WINTER",
    "rgb_to_lab",
    "lab_distance",
    "extract_dominant_color",
    "classify_hue",
    "classify_chroma",
    "classify_value",
    "classify_contrast",
]
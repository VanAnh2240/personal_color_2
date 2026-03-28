# src/models/__init__.py
from .system_1_deeplabv3 import DeepLabV3
from .system_2_clipunet  import ClipUNet

__all__ = ["DeepLabV3", "ClipUNet"]

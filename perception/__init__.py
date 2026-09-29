"""感知模块：从图像到结构化信息"""

from .types import BBox, CameraOCRResult, OCRItem
from .manager import PerceptionManager
from .sticker_classifier import StickerClassifier

__all__ = [
    "BBox",
    "CameraOCRResult",
    "OCRItem",
    "PerceptionManager",
    "StickerClassifier",
]

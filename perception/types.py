"""感知模块数据结构定义"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBox:
    """边界框"""

    x1: float  #: 左上角 x
    y1: float  #: 左上角 y
    x2: float  #: 右下角 x
    y2: float  #: 右下角 y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float) -> bool:
        """判断点 (x, y) 是否在 bbox 内"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class OCRItem:
    """单条 OCR 识别结果"""

    text: str  #: 识别文字
    bbox: BBox  #: 文字位置
    confidence: float  #: 置信度

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center


@dataclass
class CameraOCRResult:
    """单相机 OCR 结果"""

    camera_id: str  #: 相机 ID
    items: list[OCRItem] = field(default_factory=list)  #: OCR 识别条目列表

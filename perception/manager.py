"""感知管理器

对相机帧执行 OCR 识别，返回结构化结果。
支持 PaddleOCR 和模拟 OCR 两种模式。
"""

import logging
import re
import time
from typing import Optional

import numpy as np

from .types import BBox, CameraOCRResult, OCRItem

logger = logging.getLogger(__name__)


class _DummyOCR:
    """模拟 OCR 引擎，用于无 PaddleOCR 环境时的测试。

    接口与 PaddleOCR 对齐：必须提供 ``ocr()``，否则主循环首帧会直接崩溃并断开场景。
    """

    def ocr(self, image: np.ndarray, cls: bool = True):
        # 与 PaddleOCR 空结果格式兼容：无文字时返回 [None] / None
        return [None]

    def run(self, img: np.ndarray) -> tuple:
        return None, None


class PerceptionManager:
    """感知管理器，负责对相机帧执行 OCR

    :param use_real_ocr: 是否使用真实 OCR 引擎（PaddleOCR），False 则使用模拟引擎
    :param ocr_conf_threshold: OCR 结果置信度阈值，低于此值的结果被过滤
    """

    def __init__(self, use_real_ocr: bool = True, ocr_conf_threshold: float = 0.5) -> None:
        self._use_real_ocr = use_real_ocr
        self._ocr_conf_threshold = ocr_conf_threshold
        self._ocr_engine = None

    def _init_ocr(self):
        """延迟初始化 OCR 引擎"""
        if self._ocr_engine is not None:
            return
        if self._use_real_ocr:
            try:
                from paddleocr import PaddleOCR

                self._ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                logger.info("PaddleOCR 引擎初始化成功")
            except ImportError:
                logger.warning("PaddleOCR 未安装，回退到模拟 OCR；请运行: pip install paddleocr")
                self._ocr_engine = _DummyOCR()
            except Exception as e:
                logger.warning("PaddleOCR 初始化失败: %s，回退到模拟 OCR", e)
                self._ocr_engine = _DummyOCR()
        else:
            self._ocr_engine = _DummyOCR()

    def _run_ocr(self, image: np.ndarray) -> list[OCRItem]:
        """对单张图像执行 OCR

        :param image: 输入图像 (BGR 格式)
        :returns: OCR 识别结果列表
        """
        self._init_ocr()
        items: list[OCRItem] = []

        result = self._ocr_engine.ocr(image, cls=True)
        if result is None or not result or result[0] is None:
            return items

        # PaddleOCR 返回格式: [ [[box, [text, conf]], ...] ]
        try:
            # result[0] 是单张图的结果列表
            for line in result[0]:
                if line is None:
                    continue
                # line 格式: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence) ]
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    box_points, text_info = line
                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                        text, conf = text_info
                        print(f"[OCR] 识别到文字: '{text}' (置信度: {conf:.2f})")
                        logger.info("OCR 识别到文字: '%s' (置信度: %.2f)", text, conf)
                    else:
                        continue
                else:
                    continue

                if conf < self._ocr_conf_threshold:
                    continue

                # 从四点坐标提取 bbox
                if isinstance(box_points, (list, np.ndarray)) and len(box_points) >= 2:
                    xs = [p[0] for p in box_points]
                    ys = [p[1] for p in box_points]
                    bbox = BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
                else:
                    continue

                items.append(OCRItem(text=text, bbox=bbox, confidence=float(conf)))
        except Exception as e:
            logger.warning("OCR 结果解析异常: %s", e)

        return items

    def process(self, frames: list, timestamp: Optional[float] = None) -> dict:
        """对相机帧执行 OCR 处理

        :param frames: CameraFrame 列表，每个元素有 .id 和 .frame
        :param timestamp: 可选时间戳
        :returns: {'ocr': {camera_id: CameraOCRResult, ...}}
        """
        if timestamp is None:
            timestamp = time.time()

        ocr_results: dict[str, CameraOCRResult] = {}
        for frame in frames:
            items = self._run_ocr(frame.frame)
            ocr_results[frame.id] = CameraOCRResult(camera_id=frame.id, items=items)
            logger.debug("相机 %s OCR 识别到 %d 条文字", frame.id, len(items))

        return {"ocr": ocr_results}

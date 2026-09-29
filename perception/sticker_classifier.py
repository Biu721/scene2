"""贴纸/车身颜色分类器（场景 C 专用）

场景 C 仅三种车：
- 红色轿车：深酒红 OpenCV HSV ≈ (4, 210, 64)，白字 ≈ (0, 12, 236)
- 蓝色轿车：亮蓝 OpenCV HSV ≈ (107, 234, 207)，白字 ≈ (102, 14, 241)
- 深绿三轮：车身深绿 OpenCV HSV ≈ (62, 204, 103)

注意：所有车顶都有黄色车牌，黄色一律忽略，不参与分类与三轮判定。
返回：红色 / 蓝色 / 绿色 / unknown
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# OpenCV HSV：H 0-180, S 0-255, V 0-255
_COLOR_RANGES = {
    # 深酒红车身 (4, 210, 64)：V 只有约 64，下限必须低于 120
    # 同时覆盖高亮正红 (0, 218, 210)
    "红色1": ((0, 150, 40), (12, 255, 255)),
    "红色2": ((168, 150, 40), (180, 255, 255)),
    # 亮蓝车身 (107, 234, 207)
    "蓝色": ((100, 160, 120), (116, 255, 255)),
    # 三轮深绿 (62, 204, 103)
    "绿色": ((50, 120, 40), (78, 255, 180)),
}

# 车顶黄牌：排除用，不参与投票
_YELLOW_PLATE_RANGE = ((18, 80, 80), (40, 255, 255))

_COLOR_NAMES = ["蓝色", "红色", "绿色"]


class StickerClassifier:
    """场景 C 颜色分类器（红 / 蓝 / 绿，忽略黄牌）"""

    def __init__(self, min_pixel_ratio: float = 0.05) -> None:
        self._min_pixel_ratio = min_pixel_ratio

    def classify_from_vehicle_crop(
        self,
        vehicle_crop: np.ndarray,
        sticker_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> str:
        """从裁剪图识别主色调。返回 红色 / 蓝色 / 绿色 / unknown。"""
        ratios = self.color_ratios(vehicle_crop, sticker_region)
        if not ratios:
            return "unknown"

        red = ratios.get("红色", 0.0)
        blue = ratios.get("蓝色", 0.0)
        green = ratios.get("绿色", 0.0)

        scores = {"红色": red, "蓝色": blue, "绿色": green}
        best = max(scores, key=scores.get)
        best_score = scores[best]

        if best_score < self._min_pixel_ratio:
            return "unknown"

        logger.info(
            "颜色分类: %s (红=%.1f%% 蓝=%.1f%% 绿=%.1f%%，黄牌已忽略)",
            best,
            red * 100,
            blue * 100,
            green * 100,
        )
        return best

    def color_ratios(
        self,
        vehicle_crop: np.ndarray,
        sticker_region: Optional[Tuple[float, float, float, float]] = None,
    ) -> dict[str, float]:
        """返回红/蓝/绿在有效像素中的占比（黄牌像素已排除）。"""
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {}

        try:
            import cv2

            h, w = vehicle_crop.shape[:2]
            if sticker_region is not None:
                x1 = int(sticker_region[0] * w)
                y1 = int(sticker_region[1] * h)
                x2 = int(sticker_region[2] * w)
                y2 = int(sticker_region[3] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return {}
                roi = vehicle_crop[y1:y2, x1:x2]
            else:
                roi = vehicle_crop

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            if total_pixels == 0:
                return {}

            # 排除：低饱和、过暗、车顶黄牌
            text_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 40, 255]))
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 35]))
            y_lo, y_hi = _YELLOW_PLATE_RANGE
            yellow_mask = cv2.inRange(hsv, np.array(y_lo), np.array(y_hi))
            exclude_mask = cv2.bitwise_or(text_mask, dark_mask)
            exclude_mask = cv2.bitwise_or(exclude_mask, yellow_mask)
            valid_mask = cv2.bitwise_not(exclude_mask)
            valid_count = cv2.countNonZero(valid_mask)

            if valid_count < total_pixels * 0.05:
                # 黄牌占比过大时，仍排除黄色后再回退
                valid_mask = cv2.bitwise_not(yellow_mask)
                valid_count = cv2.countNonZero(valid_mask)
                if valid_count < total_pixels * 0.02:
                    return {}

            ratios: dict[str, float] = {}
            for color_name in _COLOR_NAMES:
                if color_name == "红色":
                    lower1, upper1 = _COLOR_RANGES["红色1"]
                    lower2, upper2 = _COLOR_RANGES["红色2"]
                    mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
                    mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    lower, upper = _COLOR_RANGES[color_name]
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

                masked = cv2.bitwise_and(mask, valid_mask)
                count = cv2.countNonZero(masked)
                ratios[color_name] = count / max(valid_count, 1)

            return ratios

        except ImportError:
            logger.warning("OpenCV 未安装，无法进行颜色分类")
            return {}
        except Exception as e:
            logger.warning("颜色分类异常: %s", e)
            return {}

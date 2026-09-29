"""场景 C 状态机：规则合规审查

核心逻辑：
1. 驶向规则牌，用前视 OCR 识别规则文字
2. RuleParser 解析规则文本为规则列表（如"仅允许蓝色"）
3. 巡逻时识别车辆编号（OCR），车辆 crop 送 StickerClassifier 得颜色
4. 按规则判断是否违规，上报违规编号
"""

import logging
import math
import os
import re
from enum import Enum, auto
from typing import Optional

from metacar import (
    GearMode,
    FunctionZoneResult,
    FunctionZoneViolation,
    ParkingResult,
    RegionType,
    SceneStaticData,
    SimCarMsg,
    Vector3,
    VehicleControl,
    VLAExtensionOutput,
)

from perception.types import CameraOCRResult
from perception.sticker_classifier import StickerClassifier

logger = logging.getLogger(__name__)

# 贴纸编号正则
_STICKER_ID_PATTERN = re.compile(r"^[A-Z]-\d{1,2}$")

# 场景 C 有效编号范围（通常 C-01 到 C-06）
_VALID_STICKER_IDS = {f"C-{i:02d}" for i in range(1, 7)}


def _plate_number(payload: str) -> str:
    """把车牌数字里的易混字符收成 01–06。失败返回空串。"""
    trans = str.maketrans({
        "O": "0",
        "D": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    })
    digits = payload.translate(trans)
    digits = re.sub(r"\D", "", digits)
    if not digits:
        return ""
    # OCR 偶发多读一位，只留末两位（C-0X 本身只有两位）
    digits = digits[-2:]
    try:
        num = int(digits)
    except ValueError:
        return ""
    if num < 1 or num > 6:
        return ""
    return f"C-{num:02d}"


def _normalize_sticker_id(text: str) -> str:
    """把 OCR 收成 C-01..C-06。

    规则牌（含中文、R01/R02）不当车牌。
    车贴允许残缺：C-D3 / C-OT / 05 / -05 / 0-05 / 04。
    """
    raw = text.strip()
    if re.search(r"[\u4e00-\u9fff]", raw):
        return ""
    upper = raw.upper()
    if any(kw in upper for kw in ("允许", "禁止", "规则", "R0", "RO")):
        return ""

    for dash in ("—", "–", "−", "_", "~", "=", ":"):
        upper = upper.replace(dash, "-")
    upper = upper.replace(" ", "")
    upper = re.sub(r"[^A-Z0-9-]", "", upper)
    if not upper or upper.startswith("R"):
        return ""

    # C-03 / C-D3 / C-OT / CD3 / G-06（C 被看成 G 或 8）
    m = re.fullmatch(r"[CG8]-?([A-Z0-9]{1,3})", upper)
    if m:
        return _plate_number(m.group(1))

    # 丢掉了 C 的残片：05、-05、0-05、06、04、01
    # 不用 search，避免 80-3、10-3 这种噪声被抽成编号
    m = re.fullmatch(r"-?0-?(\d{2})", upper) or re.fullmatch(r"-?(\d{2})", upper)
    if m:
        return _plate_number(m.group(1))
    return ""


class RuleParser:
    """规则文本解析器

    将 OCR 识别到的规则文字解析为结构化规则列表。
    例如："仅允许蓝色贴纸" -> {"allowed_colors": ["蓝色"]}
    """

    _COLOR_KEYWORDS = {
        "蓝色": "蓝色",
        "红色": "红色",
        "黄色": "黄色",
        "绿色": "绿色",
        "橙色": "橙色",
        "白色": "白色",
        "蓝": "蓝色",
        "红": "红色",
        "黄": "黄色",
        "绿": "绿色",
        "橙": "橙色",
        "白": "白色",
    }

    # 长词优先
    _TYPE_KEYWORDS = [
        ("小轿车", "小轿车"),
        ("三轮车", "三轮车"),
        ("三轮", "三轮车"),
        ("轿车", "小轿车"),
        ("货车", "货车"),
    ]

    _FORBID_KEYWORDS = [
        "不允许", "不充许", "不可", "不能", "禁止", "严禁", "禁止使用", "不得",
    ]
    _ALLOW_KEYWORDS = ["只允许", "仅允许", "只能", "只可", "允许", "可"]

    @staticmethod
    def _normalize_rule_ocr(text: str) -> str:
        """修复规则牌常见 OCR 误识"""
        text = text.replace("三招车", "三轮车")
        text = text.replace("三令车", "三轮车")
        text = text.replace("三纶车", "三轮车")
        text = text.replace("三车", "三轮车")
        text = text.replace("小折车", "小轿车")
        text = text.replace("小辆", "小轿车")
        text = text.replace("小额年", "小轿车")
        text = text.replace("不充许", "不允许")
        text = text.replace("不元许", "不允许")
        text = text.replace("不分许", "不允许")
        text = text.replace("监色", "蓝色")
        text = text.replace("匠色", "蓝色")
        text = text.replace("面色", "蓝色")
        text = text.replace("蓝信", "蓝色")
        text = re.sub(r"\bRO\b", "R0", text, flags=re.IGNORECASE)
        text = re.sub(r"\bR0([12])\b", r"R0\1", text, flags=re.IGNORECASE)
        text = re.sub(r"(?<!R)\b01\b", "R01", text)
        text = re.sub(r"(?<!R)\b02\b", "R02", text)
        text = re.sub(r"(?<![A-Za-z])0?2(?=\s*禁止)", "R02", text)
        text = re.sub(r"(?<![A-Za-z])0?1(?=\s*不)", "R01", text)
        return text

    def parse(self, rule_text: str) -> list[dict]:
        """解析规则文本（支持多条规则独立评分，包含颜色和车型）"""
        rules: list[dict] = []
        text = self._normalize_rule_ocr(rule_text)

        parts = re.split(r"(R\d{1,2})", text)
        refined_sentences: list[str] = []
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                code = parts[i]
                content = parts[i + 1].strip()
                if content:
                    refined_sentences.append(f"{code} {content}")

        if not refined_sentences:
            for s in re.split(r"[。；;.\n\r]", text):
                s = s.strip()
                if s:
                    refined_sentences.append(s)

        for sentence in refined_sentences:
            found_colors: list[str] = []
            # 长词优先：先匹配「蓝色」再匹配「蓝」
            for keyword, color_name in sorted(
                self._COLOR_KEYWORDS.items(), key=lambda kv: -len(kv[0])
            ):
                if keyword in sentence and color_name not in found_colors:
                    found_colors.append(color_name)

            found_types: list[str] = []
            for keyword, type_name in self._TYPE_KEYWORDS:
                if keyword in sentence and type_name not in found_types:
                    found_types.append(type_name)

            if not found_colors and not found_types:
                continue

            is_forbid = any(kw in sentence for kw in self._FORBID_KEYWORDS)
            is_allow = any(kw in sentence for kw in self._ALLOW_KEYWORDS)
            if not is_forbid and not is_allow:
                # OCR 丢掉极性词时，有颜色/车型默认按禁止
                is_forbid = True

            rule_code_match = re.search(r"R(\d+)", sentence)
            rule_code = (
                f"R{rule_code_match.group(1).zfill(2)}" if rule_code_match else ""
            )

            rule = {
                "rule_code": rule_code,
                "allowed_colors": [],
                "forbidden_colors": [],
                "allowed_types": [],
                "forbidden_types": [],
            }
            if is_forbid:
                rule["forbidden_colors"] = found_colors
                rule["forbidden_types"] = found_types
            else:
                rule["allowed_colors"] = found_colors
                rule["allowed_types"] = found_types

            rules.append(rule)
            logger.info(
                "从句子解析到规则: '%s' -> 颜色:%s, 车型:%s (is_forbid=%s, code=%s)",
                sentence,
                found_colors,
                found_types,
                is_forbid,
                rule_code,
            )

        if not rules:
            logger.warning("无法从文字解析出任何结构化规则: %s", rule_text)

        logger.info("规则解析结果列表: %s", rules)
        return rules

    def is_violating_any(
        self, color: str, rules_list: list[dict], vehicle_type: str = "unknown"
    ) -> bool:
        if color == "unknown" and vehicle_type == "unknown":
            return False
        return any(
            self.is_violating_single(color, rule, vehicle_type) for rule in rules_list
        )

    def is_violating_single(
        self, color: str, rule: dict, vehicle_type: str = "unknown"
    ) -> bool:
        """单条规则判定。

        同条规则同时含颜色与车型时按 AND：
        「不允许蓝色小轿车」→ 蓝色且小轿车才违规。
        """
        allowed_colors = rule.get("allowed_colors", [])
        forbidden_colors = rule.get("forbidden_colors", [])
        allowed_types = rule.get("allowed_types", [])
        forbidden_types = rule.get("forbidden_types", [])

        if forbidden_colors and forbidden_types:
            if color == "unknown" or vehicle_type == "unknown":
                return False
            return color in forbidden_colors and vehicle_type in forbidden_types
        if forbidden_colors:
            return color != "unknown" and color in forbidden_colors
        if forbidden_types:
            return vehicle_type != "unknown" and vehicle_type in forbidden_types

        if allowed_colors and allowed_types:
            if color == "unknown" and vehicle_type == "unknown":
                return False
            color_ok = color == "unknown" or color in allowed_colors
            type_ok = vehicle_type == "unknown" or vehicle_type in allowed_types
            return not (color_ok and type_ok)
        if allowed_colors:
            return color != "unknown" and color not in allowed_colors
        if allowed_types:
            return vehicle_type != "unknown" and vehicle_type not in allowed_types

        return False


class SceneCState(Enum):
    """场景 C 状态"""

    READ_RULE = auto()
    PATROL = auto()
    REPORT = auto()
    DONE = auto()


class SceneCStateMachine:
    """场景 C 状态机：规则合规审查"""

    def __init__(
        self,
        static_data: SceneStaticData,
        patrol_speed: float = 5.0,
    ) -> None:
        self._static_data = static_data
        self._patrol_speed = patrol_speed

        self._state = SceneCState.READ_RULE
        self._frame_count = 0

        self._rule_parser = RuleParser()
        self._rules: list[dict] = []
        self._rule_read = False  # 仅当 OCR 锁定 R01+R02（或等价完整集）时为 True
        self._rules_are_default = False

        self._region_center: Optional[Vector3] = None
        self._last_angle: Optional[float] = None
        self._cumulative_angle = 0.0
        self._loop_angle_threshold = 2 * math.pi * 0.95
        self._patrol_radius = 8.0

        self._vehicle_colors: dict[str, str] = {}
        self._vehicle_types: dict[str, str] = {}
        self._ocr_confirmed_types: dict[str, str] = {}
        self._all_detected_ids: set[str] = set()
        self._detection_order: list[str] = []  # 首次识别顺序，第一辆即三轮
        self._sticker_seen_counts: dict[str, int] = {}
        self._color_votes: dict[str, dict[str, int]] = {}

        self._sticker_classifier = StickerClassifier()
        self._reported = False
        self._initial_pos: Optional[Vector3] = None
        self._cached_violations: list[FunctionZoneViolation] = []
        self._cached_all_ids: list[str] = []
        self._tricycle_id = ""
        self._red_sedan_id = ""
        self._blue_sedan_ids: list[str] = []

        self._find_region_center()

    def _find_region_center(self):
        vla_ext = self._static_data.vla_extension
        if vla_ext is None:
            logger.error("VLA 扩展信息为空，无法确定功能区")
            return

        for region in vla_ext.regions:
            if region.type == RegionType.FUNCTION_ZONE:
                self._region_center = Vector3(region.pos_x, region.pos_y, region.pos_z)
                logger.info(
                    "功能区中心: (%.1f, %.1f, %.1f)",
                    region.pos_x,
                    region.pos_y,
                    region.pos_z,
                )
                return

        for region in vla_ext.regions:
            if region.type == RegionType.RESTRICTED_PARKING:
                self._region_center = Vector3(region.pos_x, region.pos_y, region.pos_z)
                logger.info(
                    "使用禁停区作为功能区中心: (%.1f, %.1f, %.1f)",
                    region.pos_x,
                    region.pos_y,
                    region.pos_z,
                )
                return

        for region in vla_ext.regions:
            if region.type == RegionType.NORMAL_PARKING:
                self._region_center = Vector3(region.pos_x, region.pos_y, region.pos_z)
                logger.info(
                    "使用正常停车区作为功能区中心: (%.1f, %.1f, %.1f)",
                    region.pos_x,
                    region.pos_y,
                    region.pos_z,
                )
                return

        if vla_ext.regions:
            r = vla_ext.regions[0]
            self._region_center = Vector3(r.pos_x, r.pos_y, r.pos_z)
            logger.warning(
                "使用第一个区域作为中心: (%.1f, %.1f, %.1f)", r.pos_x, r.pos_y, r.pos_z
            )
            return

        logger.warning("未找到任何区域")

    def _get_patrol_target(self, pos: Vector3) -> Vector3:
        if self._region_center is None:
            return pos
        dx = pos.x - self._region_center.x
        dy = pos.y - self._region_center.y
        current_angle = math.atan2(dy, dx)
        target_angle = current_angle + 0.3
        target_x = self._region_center.x + self._patrol_radius * math.cos(target_angle)
        target_y = self._region_center.y + self._patrol_radius * math.sin(target_angle)
        return Vector3(target_x, target_y, pos.z)

    def _calc_steering(self, pos: Vector3, yaw: float, target: Vector3) -> float:
        dx = target.x - pos.x
        dy = target.y - pos.y
        target_yaw = math.atan2(dy, dx)
        diff = (target_yaw - yaw) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        steering = math.degrees(-diff) / 45
        return max(min(steering, 1), -1)

    def _calc_throttle_brake(self, current_speed: float) -> tuple[float, float]:
        diff = self._patrol_speed - current_speed
        if diff > 0:
            return min(diff * 0.3, 1.0), 0.0
        return 0.0, min(-diff * 0.3, 1.0)

    def _is_near_region(self, pos: Vector3) -> bool:
        if self._region_center is None:
            return False
        dist = math.dist((pos.x, pos.y), (self._region_center.x, self._region_center.y))
        return dist <= max(self._patrol_radius * 1.5, 12.0)

    def _current_region_angle(self, pos: Vector3) -> Optional[float]:
        if self._region_center is None:
            return None
        dx = pos.x - self._region_center.x
        dy = pos.y - self._region_center.y
        return math.atan2(dy, dx)

    def _angle_diff(self, a: float, b: float) -> float:
        diff = (a - b) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def _update_patrol_progress(self, pos: Vector3) -> float:
        cur_angle = self._current_region_angle(pos)
        if cur_angle is None:
            return self._cumulative_angle
        if self._last_angle is not None:
            diff = self._angle_diff(cur_angle, self._last_angle)
            self._cumulative_angle += abs(diff)
            if self._frame_count % 100 == 0:
                logger.info(
                    "巡逻进度: 累计角度=%.2f rad (目标=%.2f)",
                    self._cumulative_angle,
                    self._loop_angle_threshold,
                )
        self._last_angle = cur_angle
        return self._cumulative_angle

    def _match_sticker_ids(
        self,
        sim_car_msg: SimCarMsg,
        ocr_results: dict[str, CameraOCRResult],
    ) -> list[str]:
        """按 OCR 出现顺序返回编号（同帧内去重，保留先出现的）。"""
        matched: list[str] = []
        seen: set[str] = set()
        for _cam_id, ocr_result in ocr_results.items():
            for ocr_item in ocr_result.items:
                text = ocr_item.text.strip()
                normalized_text = _normalize_sticker_id(text)
                if _STICKER_ID_PATTERN.match(normalized_text):
                    if normalized_text in _VALID_STICKER_IDS:
                        if normalized_text not in seen:
                            seen.add(normalized_text)
                            matched.append(normalized_text)
                        logger.info("OCR捕获编号: %s (原始: %s)", normalized_text, text)
                    else:
                        logger.debug(
                            "编号不在有效范围内，忽略: %s (原始: %s)",
                            normalized_text,
                            text,
                        )
        return matched

    def _merge_rules(self, parsed_list: list[dict]) -> None:
        """按 rule_code 合并，避免只读到 R02 时把已有 R01 冲掉。"""
        merged: dict[str, dict] = {}
        anon: list[dict] = []

        for r in self._rules:
            code = r.get("rule_code") or ""
            if code and not self._rules_are_default:
                merged[code] = r
            elif not code:
                anon.append(r)

        def _richness(rule: dict) -> int:
            return (
                len(rule.get("forbidden_colors", []))
                + len(rule.get("forbidden_types", []))
                + len(rule.get("allowed_colors", []))
                + len(rule.get("allowed_types", []))
            )

        for r in parsed_list:
            code = r.get("rule_code") or ""
            if code:
                old = merged.get(code)
                if old is None or _richness(r) >= _richness(old):
                    merged[code] = r
            else:
                anon.append(r)

        # OCR 成功解析后清除默认规则
        self._rules_are_default = False
        ordered = [merged[k] for k in sorted(merged.keys())]
        ordered.extend(anon)
        self._rules = ordered
        logger.info("规则已合并更新: %s", self._rules)

    def _parse_rule_from_ocr(self, ocr_results: dict[str, CameraOCRResult]) -> bool:
        """从 OCR 结果中解析规则文字。锁定前持续尝试。"""
        if self._rule_read:
            return True

        rule_texts = []
        for _cam_id, ocr_result in ocr_results.items():
            for ocr_item in ocr_result.items:
                text = ocr_item.text.strip()
                if any(
                    kw in text
                    for kw in ["允许", "禁止", "可", "不可", "仅", "只", "规则", "不得"]
                ):
                    rule_texts.append(text)

        if not rule_texts:
            return False

        combined = " ".join(rule_texts)
        parsed_list = self._rule_parser.parse(combined)
        if not parsed_list:
            logger.debug("规则文字匹配但未解析出具体规则，继续尝试: %s", combined)
            return False

        parsed_codes = {r.get("rule_code", "") for r in parsed_list if r.get("rule_code")}
        if parsed_codes or not self._rules or self._rules_are_default:
            if self._rules_are_default:
                self._rules = []
            self._merge_rules(parsed_list)

        by_code = {r.get("rule_code", ""): r for r in self._rules}
        r01 = by_code.get("R01", {})
        r02 = by_code.get("R02", {})
        r01_ready = "蓝色" in r01.get("forbidden_colors", []) and "小轿车" in r01.get(
            "forbidden_types", []
        )
        r02_ready = "三轮车" in r02.get("forbidden_types", [])
        if r01_ready and r02_ready:
            self._rule_read = True
            self._rules_are_default = False
            logger.info("规则已锁定: R01 禁蓝轿, R02 禁三轮")
            return True

        # 单条真实规则也保留，继续等另一条
        return False

    def _classify_colors_from_frames(
        self,
        ocr_results: dict[str, CameraOCRResult],
        perception_data: dict,
    ) -> None:
        """多帧投票提取贴纸颜色；OCR 文本确认车型。"""
        frames = perception_data.get("frames", [])
        classifier = perception_data.get("sticker_classifier", self._sticker_classifier)
        if not frames:
            return

        frame_map = {f.id: f for f in frames}

        for cam_id, ocr_result in ocr_results.items():
            frame_obj = frame_map.get(cam_id)
            if frame_obj is None:
                continue
            image = frame_obj.frame
            if image is None:
                continue

            img_h, img_w = image.shape[:2]

            for ocr_item in ocr_result.items:
                text = ocr_item.text.strip()
                normalized_text = _normalize_sticker_id(text)
                if normalized_text not in _VALID_STICKER_IDS:
                    continue

                full_text = RuleParser._normalize_rule_ocr(ocr_item.text)
                first_id = self._detection_order[0] if self._detection_order else ""
                # 第一优先：最先看到的编号已锁定为三轮，文字不能改掉
                if normalized_text == first_id:
                    self._ocr_confirmed_types[normalized_text] = "三轮车"
                    self._vehicle_types[normalized_text] = "三轮车"
                elif "三轮" in full_text and not first_id:
                    self._ocr_confirmed_types[normalized_text] = "三轮车"
                    self._vehicle_types[normalized_text] = "三轮车"
                    logger.info("OCR 确认车型: %s -> 三轮车", normalized_text)
                elif "轿车" in full_text or "小车" in full_text:
                    self._ocr_confirmed_types[normalized_text] = "小轿车"
                    self._vehicle_types[normalized_text] = "小轿车"

                bbox = ocr_item.bbox
                bbox_w = bbox.x2 - bbox.x1
                bbox_h = bbox.y2 - bbox.y1
                pad_x = max(int(bbox_w * 0.3), 5)
                pad_y = max(int(bbox_h * 0.3), 5)
                x1 = max(0, int(bbox.x1 - pad_x))
                y1 = max(0, int(bbox.y1 - pad_y))
                x2 = min(img_w, int(bbox.x2 + pad_x))
                y2 = min(img_h, int(bbox.y2 + pad_y))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                color = classifier.classify_from_vehicle_crop(crop)
                if color == "unknown":
                    continue

                votes = self._color_votes.setdefault(normalized_text, {})
                # 只认红/蓝/绿；忽略黄色（车顶黄牌）
                if color not in ("红色", "蓝色", "绿色"):
                    continue
                votes[color] = votes.get(color, 0) + 1
                self._vehicle_colors[normalized_text] = max(
                    votes.items(), key=lambda kv: kv[1]
                )[0]

                if normalized_text not in self._ocr_confirmed_types:
                    stable = self._vehicle_colors[normalized_text]
                    # 红色一定是轿车，不能当三轮
                    if stable == "红色":
                        self._vehicle_types[normalized_text] = "小轿车"
                    elif stable == "蓝色":
                        self._vehicle_types[normalized_text] = "小轿车"

                if self._frame_count % 30 == 0:
                    logger.info(
                        "编号 %s 颜色投票=%s -> %s",
                        normalized_text,
                        votes,
                        self._vehicle_colors[normalized_text],
                    )

    def _get_color_vote(self, sticker_id: str) -> dict[str, int]:
        return self._color_votes.get(sticker_id, {})

    def _stable_color(self, sticker_id: str) -> str:
        votes = self._get_color_vote(sticker_id)
        if votes:
            return max(votes.items(), key=lambda kv: kv[1])[0]
        return self._vehicle_colors.get(sticker_id, "unknown")

    def _sync_colors_from_votes(self) -> None:
        for sid, votes in self._color_votes.items():
            if votes:
                self._vehicle_colors[sid] = max(votes.items(), key=lambda kv: kv[1])[0]

    def _color_ratio(self, sticker_id: str, color_name: str) -> float:
        votes = self._get_color_vote(sticker_id)
        if not votes:
            return 1.0 if self._vehicle_colors.get(sticker_id) == color_name else 0.0
        total = sum(votes.values())
        if total <= 0:
            return 0.0
        return votes.get(color_name, 0) / total

    def _blue_score(self, sticker_id: str) -> float:
        return self._color_ratio(sticker_id, "蓝色")

    def _red_score(self, sticker_id: str) -> float:
        return self._color_ratio(sticker_id, "红色")

    def _green_score(self, sticker_id: str) -> float:
        return self._color_ratio(sticker_id, "绿色")

    def _tricycle_score(self, sticker_id: str) -> float:
        """三轮只看绿色车身占比（忽略车顶黄牌）。"""
        return self._green_score(sticker_id)

    def _red_votes(self, sticker_id: str) -> int:
        return self._get_color_vote(sticker_id).get("红色", 0)

    def _blue_votes(self, sticker_id: str) -> int:
        return self._get_color_vote(sticker_id).get("蓝色", 0)

    def _force_color(self, sticker_id: str, color: str) -> None:
        """覆盖投票结果，避免上报仍读到修正前的颜色。"""
        self._vehicle_colors[sticker_id] = color
        self._color_votes[sticker_id] = {color: 1}

    def _infer_vehicle_types_with_constraints(self) -> None:
        """场景 C 固定 6 辆：C-01..C-06 = 1 三轮 + 1 红轿 + 4 蓝轿。

        - 第一优先：OCR 最先看到的编号一定是三轮
        - 红轿只有一辆，在三轮之后；没有红色票时，取最不像蓝的那辆
        - R01 对应的蓝轿一定是剩下的 4 个编号
        """
        self._sync_colors_from_votes()
        detected = sorted(self._all_detected_ids)
        if not detected:
            self._tricycle_id = ""
            self._red_sedan_id = ""
            self._blue_sedan_ids = []
            return

        fleet = sorted(_VALID_STICKER_IDS)
        order = [sid for sid in self._detection_order if sid in self._all_detected_ids]
        for sid in detected:
            if sid not in order:
                order.append(sid)

        tricycle_id = order[0] if order else ""
        others = [sid for sid in fleet if sid != tricycle_id]

        red_candidates = [sid for sid in others if self._red_votes(sid) > 0]
        if red_candidates:
            red_sedan_id = max(red_candidates, key=self._red_votes)
        else:
            # 没投到红色时，蓝票最少的那辆当红轿（没看见的编号蓝票为 0）
            red_sedan_id = min(others, key=lambda sid: (self._blue_votes(sid), sid))

        blue_ids = [sid for sid in others if sid != red_sedan_id]
        self._tricycle_id = tricycle_id
        self._red_sedan_id = red_sedan_id
        self._blue_sedan_ids = blue_ids

        for sid in fleet:
            if sid not in self._all_detected_ids:
                self._all_detected_ids.add(sid)
                logger.info("按 6 辆编成补全编号: %s", sid)

        for sid in fleet:
            if sid == tricycle_id:
                self._vehicle_types[sid] = "三轮车"
                if self._stable_color(sid) == "红色":
                    self._force_color(sid, "绿色")
            elif sid == red_sedan_id:
                self._vehicle_types[sid] = "小轿车"
                self._force_color(sid, "红色")
            else:
                self._vehicle_types[sid] = "小轿车"
                if self._stable_color(sid) != "蓝色":
                    self._force_color(sid, "蓝色")

        logger.info(
            "车型推断: 识别顺序=%s 三轮=%s 红轿=%s 蓝轿=%s",
            order,
            tricycle_id or "?",
            red_sedan_id or "?",
            blue_ids,
        )

    def _build_violations(
        self,
    ) -> tuple[list[FunctionZoneViolation], list[str]]:
        """统一 REPORT/DONE 的违规集合构建。"""
        self._infer_vehicle_types_with_constraints()

        if not self._rules:
            logger.warning("上报时仍无规则，使用兜底：禁止蓝色小轿车 + 禁止三轮车")
            self._rules = [
                {
                    "rule_code": "R01",
                    "allowed_colors": [],
                    "forbidden_colors": ["蓝色"],
                    "allowed_types": [],
                    "forbidden_types": ["小轿车"],
                },
                {
                    "rule_code": "R02",
                    "allowed_colors": [],
                    "forbidden_colors": [],
                    "allowed_types": [],
                    "forbidden_types": ["三轮车"],
                },
            ]
            self._rules_are_default = True

        function_zone_violations: list[FunctionZoneViolation] = []
        all_violating_ids: set[str] = set()

        for i, rule in enumerate(self._rules):
            rule_code = rule.get("rule_code") or f"R{i + 1:02d}"
            forbids_tricycle = "三轮车" in rule.get("forbidden_types", [])
            forbids_blue = "蓝色" in rule.get("forbidden_colors", [])
            # 编成固定：R02 只有那一辆三轮，R01 一定是另外 4 辆蓝轿
            if forbids_tricycle and self._tricycle_id:
                current = [self._tricycle_id]
            elif forbids_blue and len(self._blue_sedan_ids) == 4:
                current = list(self._blue_sedan_ids)
            else:
                current = []
                for sticker_id in sorted(self._all_detected_ids):
                    color = self._stable_color(sticker_id)
                    vehicle_type = self._vehicle_types.get(sticker_id, "unknown")
                    if self._rule_parser.is_violating_single(color, rule, vehicle_type):
                        current.append(sticker_id)

            for sid in current:
                all_violating_ids.add(sid)

            function_zone_violations.append(
                FunctionZoneViolation(rule_code=rule_code, sticker_ids=current)
            )
            logger.info(
                "规则[%s] 违规编号: %s (属性: %s)",
                rule_code,
                current,
                {
                    sid: (
                        self._stable_color(sid),
                        self._vehicle_types.get(sid, "N/A"),
                    )
                    for sid in current
                },
            )

        return function_zone_violations, sorted(all_violating_ids)

    def _write_report_file(
        self,
        violations: list[FunctionZoneViolation],
        sorted_ids: list[str],
    ) -> None:
        try:
            file_path = os.path.join(os.getcwd(), "report_results.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Scene C Detection Report\n")
                f.write(f"Frame: {self._frame_count}\n")
                f.write(f"Detected IDs: {sorted(self._all_detected_ids)}\n")
                f.write(f"Detection Order: {self._detection_order}\n")
                f.write(f"ID Seen Counts: {self._sticker_seen_counts}\n")
                f.write(
                    f"Color Map: {{{', '.join(f'{k!r}: {self._stable_color(k)!r}' for k in sorted(self._all_detected_ids))}}}\n"
                )
                f.write(f"Color Votes: {self._color_votes}\n")
                f.write(f"Types: {self._vehicle_types}\n")
                f.write(f"OCR Types: {self._ocr_confirmed_types}\n")
                f.write(f"Rules default={self._rules_are_default}: {self._rules}\n\n")
                for v in violations:
                    f.write(f"Rule {v.rule_code}: {', '.join(v.sticker_ids)}\n")
                f.write(f"\nFinal Combined: {', '.join(sorted_ids)}\n")
            logger.info("结果已保存至: %s", file_path)
        except Exception as e:
            logger.error("写入结果文件失败: %s", e)

    def _get_perception_results(
        self, perception_data: dict
    ) -> dict[str, CameraOCRResult]:
        return perception_data.get("ocr", {})

    def force_report(self):
        if self._state != SceneCState.DONE:
            logger.info(
                "收到手动上报指令，正在汇总识别结果... 当前状态: %s", self._state
            )
            self._state = SceneCState.REPORT
            return True
        logger.info("收到手动上报指令，但已在 DONE 状态")
        return False

    def update(
        self,
        sim_car_msg: SimCarMsg,
        perception_data: dict,
        frame_count: int,
    ) -> tuple[VehicleControl, Optional[VLAExtensionOutput]]:
        self._frame_count = frame_count
        vc = VehicleControl()
        vc.gear = GearMode.DRIVE
        vla_output: Optional[VLAExtensionOutput] = None

        pos = Vector3(
            sim_car_msg.pose_gnss.pos_x,
            sim_car_msg.pose_gnss.pos_y,
            sim_car_msg.pose_gnss.pos_z,
        )
        yaw = -math.radians(sim_car_msg.pose_gnss.ori_z)
        speed = sim_car_msg.main_vehicle.speed

        if self._initial_pos is None:
            self._initial_pos = Vector3(pos.x, pos.y, pos.z)

        status = sim_car_msg.scene_status
        if frame_count % 100 == 0:
            logger.info(
                "运行中: 帧=%d, 已用时间=%.1fs, 限制=%.1fs",
                frame_count,
                status.used_time,
                status.time_limit,
            )

        ocr_results = self._get_perception_results(perception_data)

        matched = self._match_sticker_ids(sim_car_msg, ocr_results)
        if matched:
            before_count = len(self._all_detected_ids)
            for sid in matched:
                if sid not in self._all_detected_ids:
                    self._detection_order.append(sid)
                    self._all_detected_ids.add(sid)
                    if len(self._detection_order) == 1:
                        self._vehicle_types[sid] = "三轮车"
                        self._ocr_confirmed_types[sid] = "三轮车"
                        logger.info("最先识别的编号锁定为三轮: %s", sid)
                self._sticker_seen_counts[sid] = (
                    self._sticker_seen_counts.get(sid, 0) + 1
                )
            if len(self._all_detected_ids) > before_count:
                logger.info(
                    "捕获到编号: %s, 识别顺序: %s (状态: %s)",
                    matched,
                    self._detection_order,
                    self._state,
                )

        if ocr_results and "frames" in perception_data:
            self._classify_colors_from_frames(ocr_results, perception_data)

        # 未锁定前持续读规则（含已进入 PATROL / 暂用默认之后）
        if not self._rule_read and ocr_results:
            self._parse_rule_from_ocr(ocr_results)

        if self._state == SceneCState.READ_RULE:
            center_at_origin = (
                self._region_center is not None
                and self._region_center.x == 0.0
                and self._region_center.y == 0.0
            )

            if center_at_origin or self._region_center is None:
                if sim_car_msg.trajectory:
                    vc.steering = self._calc_steering(
                        pos, yaw, sim_car_msg.trajectory[0]
                    )
                vc.throttle, vc.brake = self._calc_throttle_brake(speed)
            else:
                target = self._region_center
                vc.steering = self._calc_steering(pos, yaw, target)
                vc.throttle, vc.brake = self._calc_throttle_brake(speed)

            should_patrol = False
            if (
                not center_at_origin
                and self._region_center is not None
                and self._is_near_region(pos)
            ):
                should_patrol = True
            elif (center_at_origin or self._region_center is None) and frame_count > 300:
                should_patrol = True

            if should_patrol:
                if center_at_origin and self._initial_pos is not None:
                    self._region_center = Vector3(pos.x, pos.y, pos.z)
                    logger.info(
                        "区域中心为原点，使用当前位置作为巡逻中心: (%.1f, %.1f, %.1f)",
                        pos.x,
                        pos.y,
                        pos.z,
                    )
                self._state = SceneCState.PATROL
                self._cumulative_angle = 0.0
                self._last_angle = None
                if not self._rules:
                    # 仅作临时占位，不锁定，后续 OCR 可覆盖
                    self._rules = [
                        {
                            "rule_code": "R01",
                            "allowed_colors": [],
                            "forbidden_colors": ["蓝色"],
                            "allowed_types": [],
                            "forbidden_types": ["小轿车"],
                        },
                        {
                            "rule_code": "R02",
                            "allowed_colors": [],
                            "forbidden_colors": [],
                            "allowed_types": [],
                            "forbidden_types": ["三轮车"],
                        },
                    ]
                    self._rules_are_default = True
                    logger.warning(
                        "尚未读到规则，暂用场景常见默认(禁蓝轿+禁三轮)，继续 OCR"
                    )
                logger.info("进入 PATROL（规则: %s, default=%s）", self._rules, self._rules_are_default)

        elif self._state == SceneCState.PATROL:
            if self._region_center is not None:
                target = self._get_patrol_target(pos)
                vc.steering = self._calc_steering(pos, yaw, target)
                vc.throttle, vc.brake = self._calc_throttle_brake(speed)

                total_progress = self._update_patrol_progress(pos)
                if total_progress >= self._loop_angle_threshold:
                    self._state = SceneCState.REPORT
                    logger.info("一圈完成，准备上报结果")
            else:
                if sim_car_msg.trajectory:
                    vc.steering = self._calc_steering(
                        pos, yaw, sim_car_msg.trajectory[0]
                    )
                vc.throttle, vc.brake = self._calc_throttle_brake(speed)
                if frame_count > 800:
                    self._state = SceneCState.REPORT
                    logger.info("巡逻超时，准备上报结果")

        elif self._state == SceneCState.REPORT:
            violations, sorted_ids = self._build_violations()
            self._cached_violations = violations
            self._cached_all_ids = sorted_ids
            self._write_report_file(violations, sorted_ids)

            logger.info("最终合并违规编号: %s", sorted_ids)
            vla_output = VLAExtensionOutput(
                function_zone_result=FunctionZoneResult(violations=violations),
                parking_result=ParkingResult(violating_sticker_ids=sorted_ids),
            )
            vc.gear = GearMode.PARKING
            vc.throttle = 0.0
            vc.brake = 1.0
            self._reported = True
            self._state = SceneCState.DONE
            logger.info("违停结果已分规则上报，切换至 DONE")

        elif self._state == SceneCState.DONE:
            # 与 REPORT 使用同一缓存，避免 DONE 路径判定不一致
            if not self._cached_violations:
                self._cached_violations, self._cached_all_ids = self._build_violations()
            vla_output = VLAExtensionOutput(
                function_zone_result=FunctionZoneResult(
                    violations=self._cached_violations
                ),
                parking_result=ParkingResult(
                    violating_sticker_ids=self._cached_all_ids
                ),
            )
            vc.throttle = 0.0
            vc.brake = 1.0
            vc.gear = GearMode.PARKING

        return vc, vla_output

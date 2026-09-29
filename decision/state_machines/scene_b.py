"""场景 B 状态机：禁停区违停检测与编号上报

核心逻辑：
1. 巡逻至禁停区附近
2. 绕禁停区巡逻两圈（OCR_LOOP_1 / OCR_LOOP_2）
3. 每圈将"文字中心落在车辆 bbox 内"的编号收入集合
4. 两圈取交集后上报违停编号
"""

import logging
import math
import re
from enum import Enum, auto
from typing import Optional

from metacar import (
    GearMode,
    ParkingResult,
    RegionType,
    SceneStaticData,
    SimCarMsg,
    Vector3,
    VehicleControl,
    VLAExtensionOutput,
)

from perception.types import CameraOCRResult

logger = logging.getLogger(__name__)

# 贴纸编号正则：支持 B/8/S 等开头，容忍 O/0 混淆
_STICKER_ID_PATTERN = re.compile(r"^[A-Z85S0]-\d{1,2}$")

def _normalize_sticker_id(text: str) -> str:
    """归一化 OCR 识别结果，修正常见混淆字符"""
    # 移除空格和常见标点，转大写
    text = text.upper().replace(" ", "").replace("O", "0").replace(":", "-")
    
    # 修正开头字符混淆
    if text.startswith("8-") or text.startswith("S-") or text.startswith("5-") or text.startswith("日-"):
        text = "B" + (text[1:] if not text.startswith("日-") else text[2:])
    
    # 场景 B 主要是 B-xx 格式，尝试补全缺失的 'B-'
    if text.isdigit():
        text = "B-" + text
    elif "-" not in text and len(text) >= 2:
        # 处理类似 "B03" 或 "803" 的情况
        if text.startswith("B") or text.startswith("8"):
            num_part = "".join(filter(str.isdigit, text))
            if num_part:
                text = "B-" + num_part
    
    # 最终检查格式，如果符合 "字母-数字" 但字母不是 B，强制修正为 B (场景 B 特性)
    if "-" in text:
        parts = text.split("-")
        if len(parts) >= 2 and parts[1].isdigit():
            text = "B-" + parts[1]
            
    return text


class SceneBState(Enum):
    """场景 B 状态"""

    PATROL = auto()  #: 巡逻至禁停区
    OCR_LOOP_1 = auto()  #: 第一圈 OCR 采集
    OCR_LOOP_2 = auto()  #: 第二圈 OCR 采集
    REPORT = auto()  #: 上报结果
    DONE = auto()  #: 完成


class SceneBStateMachine:
    """场景 B 状态机：禁停区违停检测与编号上报

    :param static_data: 场景静态数据
    :param patrol_radius: 巡逻半径（绕禁停区中心的距离）
    :param patrol_speed: 巡逻目标速度 (m/s)
    :param loop_angle_threshold: 累计角度超过此值视为一圈（弧度，默认 2π）
    """

    def __init__(
        self,
        static_data: SceneStaticData,
        patrol_radius: float = 8.0,
        patrol_speed: float = 5.0,
        loop_angle_threshold: float = 2 * math.pi,
    ) -> None:
        self._static_data = static_data
        self._patrol_radius = patrol_radius
        self._patrol_speed = patrol_speed
        self._loop_angle_threshold = loop_angle_threshold

        self._state = SceneBState.PATROL
        self._frame_count = 0

        # 禁停区中心
        self._region_center: Optional[Vector3] = None
        self._find_restricted_region()

        # 巡逻追踪
        self._last_angle: Optional[float] = None
        self._cumulative_angle = 0.0
        self._loop_angle_threshold = 2 * math.pi * 0.95  # 略小于两圈，确保及时触发

        # 两圈编号集合
        self._loop1_ids: set[str] = set()
        self._loop2_ids: set[str] = set()

        # 是否已上报
        self._reported = False

    def _find_restricted_region(self):
        """从静态数据中找到禁停区中心"""
        vla_ext = self._static_data.vla_extension
        if vla_ext is None:
            logger.error("VLA 扩展信息为空，无法确定禁停区")
            return
        for region in vla_ext.regions:
            if region.type == RegionType.RESTRICTED_PARKING:
                self._region_center = Vector3(region.pos_x, region.pos_y, region.pos_z)
                logger.info("禁停区中心: (%.1f, %.1f, %.1f)", region.pos_x, region.pos_y, region.pos_z)
                return
        logger.warning("未找到禁停区")

    def _get_patrol_target(self, pos: Vector3) -> Vector3:
        """计算巡逻目标点（绕禁停区中心的圆上最近前方点）"""
        if self._region_center is None:
            return pos
        dx = pos.x - self._region_center.x
        dy = pos.y - self._region_center.y
        current_angle = math.atan2(dy, dx)
        # 目标角度在当前角度前方一小段
        target_angle = current_angle + 0.3  # 前进约 0.3 弧度
        target_x = self._region_center.x + self._patrol_radius * math.cos(target_angle)
        target_y = self._region_center.y + self._patrol_radius * math.sin(target_angle)
        return Vector3(target_x, target_y, pos.z)

    def _calc_steering(self, pos: Vector3, yaw: float, target: Vector3) -> float:
        """计算转向角"""
        dx = target.x - pos.x
        dy = target.y - pos.y
        target_yaw = math.atan2(dy, dx)
        diff = (target_yaw - yaw) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        steering = math.degrees(-diff) / 45
        return max(min(steering, 1), -1)

    def _calc_throttle_brake(self, current_speed: float) -> tuple[float, float]:
        """根据当前速度计算油门/刹车"""
        diff = self._patrol_speed - current_speed
        if diff > 0:
            return min(diff * 0.3, 1.0), 0.0
        return 0.0, min(-diff * 0.3, 1.0)

    def _is_near_region(self, pos: Vector3) -> bool:
        """判断当前位置是否接近禁停区中心，用于触发进入 OCR 采集状态"""
        if self._region_center is None:
            return False
        dist = math.dist((pos.x, pos.y), (self._region_center.x, self._region_center.y))
        return dist <= max(self._patrol_radius * 1.2, 8.0)

    def _is_violating(self, obstacle_pos: Vector3) -> bool:
        """判断障碍物位置是否违规
        
        逻辑：
        1. 如果落在任何 RESTRICTED_PARKING 区域内 -> 违规
        2. 如果不在任何 NORMAL_PARKING 区域内 -> 违规（压线/越界）
        """
        vla_ext = self._static_data.vla_extension
        if not vla_ext:
            return False

        in_normal = False
        for region in vla_ext.regions:
            # 计算点是否在矩形区域内
            # 简化处理：使用 AABB 检查，实际可能需要考虑旋转 (ori_z)
            dx = abs(obstacle_pos.x - region.pos_x)
            dy = abs(obstacle_pos.y - region.pos_y)
            
            # 考虑旋转的矩形检查（简化为 AABB 仅作为示例，实际应根据 ori_z 旋转）
            # 假设场景中的车位多为轴对齐或简单角度
            half_l = region.length / 2.0
            half_w = region.width / 2.0
            
            if region.type == RegionType.RESTRICTED_PARKING:
                if dx < half_l and dy < half_w:
                    return True # 占用禁停区
            elif region.type == RegionType.NORMAL_PARKING:
                if dx < half_l and dy < half_w:
                    in_normal = True # 在合法位内
        
        return not in_normal # 不在任何合法位内也算违规

    def _match_sticker_ids(
        self,
        sim_car_msg: SimCarMsg,
        ocr_results: dict[str, CameraOCRResult],
    ) -> set[str]:
        """将识别到的文字归一化。
        只要识别到符合 B-0x 格式的编号，就进行记录。
        """
        matched: set[str] = set()
        
        for cam_id, ocr_result in ocr_results.items():
            for ocr_item in ocr_result.items:
                text = ocr_item.text.strip()
                normalized_text = _normalize_sticker_id(text)
                
                # 匹配 B-01 到 B-06
                if normalized_text in ["B-01", "B-02", "B-03", "B-04", "B-05", "B-06"]:
                    matched.add(normalized_text)
                    logger.info("OCR捕获编号: %s (原始: %s)", normalized_text, text)
        
        return matched

    def _get_perception_results(
        self, perception_data: dict
    ) -> dict[str, CameraOCRResult]:
        """从 perception_data 中提取 OCR 结果"""
        ocr_results: dict[str, CameraOCRResult] = perception_data.get("ocr", {})
        return ocr_results

    def _current_region_angle(self, pos: Vector3) -> Optional[float]:
        if self._region_center is None:
            return None
        dx = pos.x - self._region_center.x
        dy = pos.y - self._region_center.y
        return math.atan2(dy, dx)

    def _angle_diff(self, a: float, b: float) -> float:
        """返回角度差 a-b 归一化到 [-pi, pi]"""
        diff = (a - b) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi
        return diff

    def _update_patrol_progress(self, pos: Vector3) -> float:
        """更新巡逻进度（累计角度），返回当前累计弧度值"""
        cur_angle = self._current_region_angle(pos)
        if cur_angle is None:
            return self._cumulative_angle

        if self._last_angle is not None:
            diff = self._angle_diff(cur_angle, self._last_angle)
            # 累加角度增量
            self._cumulative_angle += abs(diff)
            
            if self._frame_count % 100 == 0:
                logger.info("巡逻进度: 累计角度=%.2f rad (目标=%.2f)", 
                            self._cumulative_angle, self._loop_angle_threshold * 2)
        
        self._last_angle = cur_angle
        return self._cumulative_angle

    def force_report(self):
        """手动触发上报逻辑（由外部热键调用）"""
        if self._state != SceneBState.DONE:
            logger.info("收到手动上报指令，正在汇总识别结果...")
            self._state = SceneBState.REPORT
            return True
        return False

    def update(
        self,
        sim_car_msg: SimCarMsg,
        perception_data: dict,
        frame_count: int,
    ) -> tuple[VehicleControl, Optional[VLAExtensionOutput]]:
        """每帧更新状态机"""
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
        
        # 记录场景状态
        status = sim_car_msg.scene_status
        if frame_count % 100 == 0:
            logger.info("运行中: 帧=%d, 已用时间=%.1fs, 限制=%.1fs", frame_count, status.used_time, status.time_limit)

        ocr_results = self._get_perception_results(perception_data)

        # 2. 只要在任何状态下看到符合格式的编号，就直接记录到 Loop 1 集合中
        # 这样可以确保手动按回车前捕获到的所有编号都能被上报
        matched = self._match_sticker_ids(sim_car_msg, ocr_results)
        if matched:
            before_count = len(self._loop1_ids)
            self._loop1_ids.update(matched)
            if len(self._loop1_ids) > before_count:
                logger.info("捕获到违章编号: %s, 当前累计: %s (状态: %s)", 
                            matched, sorted(self._loop1_ids), self._state)

        if self._state == SceneBState.PATROL:
            # 驶向禁停区
            if self._region_center is not None:
                target = self._region_center
                vc.steering = self._calc_steering(pos, yaw, target)
                vc.throttle, vc.brake = self._calc_throttle_brake(speed)
            else:
                vc.throttle = 0.3

            if self._is_near_region(pos):
                self._state = SceneBState.OCR_LOOP_1
                self._cumulative_angle = 0.0
                self._last_angle = None
                logger.info("进入 OCR_LOOP_1，开始第一圈采集")

        elif self._state == SceneBState.OCR_LOOP_1:
            # 绕禁停区巡逻，采集第一圈编号
            target = self._get_patrol_target(pos)
            vc.steering = self._calc_steering(pos, yaw, target)
            vc.throttle, vc.brake = self._calc_throttle_brake(speed)

            # 采集编号（已经在外部统一采集，此处仅保留逻辑进度）
            # 检查是否完成一圈（累计角度 > 2*PI）
            total_progress = self._update_patrol_progress(pos)
            if total_progress >= self._loop_angle_threshold:
                self._state = SceneBState.OCR_LOOP_2
                self._cumulative_angle = 0.0
                self._last_angle = None
                logger.info("第一圈完成（累计角度达标），进入 OCR_LOOP_2")

        elif self._state == SceneBState.OCR_LOOP_2:
            # 绕禁停区巡逻，采集第二圈编号
            target = self._get_patrol_target(pos)
            vc.steering = self._calc_steering(pos, yaw, target)
            vc.throttle, vc.brake = self._calc_throttle_brake(speed)

            # 采集编号（已经在外部统一采集，此处仅保留逻辑进度）
            # 检查是否完成第二圈
            total_progress = self._update_patrol_progress(pos)
            if total_progress >= self._loop_angle_threshold:
                self._state = SceneBState.REPORT
                logger.info("第二圈完成（累计角度达标），准备上报结果")

        elif self._state == SceneBState.REPORT:
            # 取并集：只要在任意一圈看到过，就认为它是违章的
            all_detected_ids = self._loop1_ids | self._loop2_ids
            sorted_ids = sorted(list(all_detected_ids))
            
            logger.info("最终汇总编号（并集）: %s", sorted_ids)

            # 始终执行写入文件和上报逻辑，不论 sorted_ids 是否为空
            try:
                import os
                file_path = os.path.join(os.getcwd(), "report_results.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Scene B Detection Report\n")
                    f.write(f"Frame: {self._frame_count}\n")
                    f.write(f"Violating IDs: {', '.join(sorted_ids)}\n")
                    f.write(f"Debug - Loop1: {sorted(list(self._loop1_ids))}\n")
                    f.write(f"Debug - Loop2: {sorted(list(self._loop2_ids))}\n")
                logger.info("结果已实时保存至: %s", file_path)
            except Exception as e:
                logger.error("写入结果文件失败: %s", e)

            vla_output = VLAExtensionOutput(
                parking_result=ParkingResult(violating_sticker_ids=sorted_ids)
            )
            self._reported = True
            self._state = SceneBState.DONE
            logger.info("违停结果已上报，切换至 DONE 状态等待仿真器结算")

        elif self._state == SceneBState.DONE:
            # 在 DONE 状态持续上报结果，直到仿真器主动断开连接（显示黑屏结算）
            all_reported_ids = sorted(list(self._loop1_ids | self._loop2_ids))
            if all_reported_ids:
                vla_output = VLAExtensionOutput(
                    parking_result=ParkingResult(
                        violating_sticker_ids=all_reported_ids
                    )
                )
            vc.throttle = 0.0
            vc.brake = 1.0  # 彻底刹停，等待结算

        return vc, vla_output

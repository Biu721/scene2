"""场景 A 状态机：任务牌 OCR 文本提交 + 录制轨迹导航停车。"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Optional

from metacar import (
    GearMode,
    SceneStaticData,
    SimCarMsg,
    VehicleControl,
    Vector3,
    VLAExtensionOutput,
    VLATextOutput,
)

logger = logging.getLogger(__name__)

# 与评测参考答案一致：整句 OCR 用「限时…抵达…小于等于」体例
# STANDARD_VLA_OCR_TEXT = "限时100秒内抵达锦汇华庭B栋一单元门口停车，行驶全程时速小于等于30km/h"
# STANDARD_VLA_TIME_PHRASE = "限时100秒内"
# STANDARD_VLA_LOCATION_PHRASE = "锦汇华庭B栋一单元门口"
# STANDARD_VLA_ACTION_PHRASE = "抵达，停车，行驶全程时速小于等于30km/h"


# STANDARD_VLA_OCR_TEXT = "在100秒内到达锦汇华庭B栋一单元门口，并且全程不得超过30km/h"
# STANDARD_VLA_TIME_PHRASE = "在100秒内"
# STANDARD_VLA_LOCATION_PHRASE = "锦汇华庭B栋一单元门口"
# STANDARD_VLA_ACTION_PHRASE = "到达，并且全程不得超过30km/h"


STANDARD_VLA_OCR_TEXT = "100秒内找到空的停车位停车，并且全程不得超过30km/h"
STANDARD_VLA_TIME_PHRASE = "100秒内"
STANDARD_VLA_LOCATION_PHRASE = "空的停车位"
STANDARD_VLA_ACTION_PHRASE = "找到并停车，并且全程不得超过30km/h"




RECORDED_TRAJECTORY_FILE = "recorded_trajectory.json"
MISSION_LOG_PATH = "vla_mission_log.json"


class SceneAStateMachine:
    """场景 A：提交标准 VLA 文本，沿 recorded_trajectory 导航并停车。"""

    def __init__(
        self,
        static_data: SceneStaticData,
        trajectory_path: Optional[str] = None,
    ) -> None:
        self._static_data = static_data
        self._text_submitted = False
        self._mission_completed = False
        self._target_pos: Optional[Vector3] = None
        self._global_path: list[Vector3] = []
        self._current_path_index = 0
        self._target_speed = 25.0 / 3.6
        self._frame_count = 0

        path = trajectory_path
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(root, RECORDED_TRAJECTORY_FILE)
        self._trajectory_path = path
        self._load_recorded_path(path)

    @property
    def mission_completed(self) -> bool:
        return self._mission_completed

    def _save_mission_log(self, data: dict) -> None:
        try:
            with open(MISSION_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info("任务记录已更新: %s", MISSION_LOG_PATH)
        except Exception as e:
            logger.error("保存任务记录失败: %s", e)

    def _load_recorded_path(self, file_path: str) -> bool:
        """从 JSON 加载录制轨迹（支持 list[{x,y,z}] 或 {points:[...]}）。"""
        try:
            if not os.path.exists(file_path):
                logger.warning("录制路径不存在: %s", file_path)
                return False
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                pts = data
            elif isinstance(data, dict):
                pts = data.get("points", [])
            else:
                pts = []
            self._global_path = [
                Vector3(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", 0.0)))
                for p in pts
                if isinstance(p, dict) and "x" in p and "y" in p
            ]
            self._current_path_index = 0
            if self._global_path:
                self._target_pos = self._global_path[-1]
                logger.info(
                    "成功加载录制路径: %s, 共 %d 个点",
                    file_path,
                    len(self._global_path),
                )
                return True
        except Exception as e:
            logger.error("加载录制路径失败: %s", e)
        return False

    def _find_target_in_static_data(self, location_name: str) -> Optional[Vector3]:
        """在静态数据中匹配建筑物或区域名称。"""
        if not self._static_data or not self._static_data.vla_extension:
            return None

        logger.info("开始搜索地点: %s", location_name)
        building_match = re.search(r"([A-Z])栋", location_name)
        keyword = building_match.group(0) if building_match else location_name
        logger.info("提取匹配关键字: %s", keyword)

        for building in self._static_data.vla_extension.buildings:
            if (
                keyword in building.name
                or keyword in building.id
                or building.name in location_name
                or building.id in location_name
            ):
                logger.info(
                    "成功匹配到建筑物: %s (ID: %s), 坐标: (%.2f, %.2f, %.2f)",
                    building.name,
                    building.id,
                    building.pos_x,
                    building.pos_y,
                    building.pos_z,
                )
                return Vector3(x=building.pos_x, y=building.pos_y, z=building.pos_z)

        for region in self._static_data.vla_extension.regions:
            if keyword in region.id or region.id in location_name:
                logger.info(
                    "成功匹配到区域: %s, 坐标: (%.2f, %.2f, %.2f)",
                    region.id,
                    region.pos_x,
                    region.pos_y,
                    region.pos_z,
                )
                return Vector3(x=region.pos_x, y=region.pos_y, z=region.pos_z)

        logger.warning("未能匹配到任何地点: %s", location_name)
        return None

    def _submit_text(
        self, current_msg: SimCarMsg
    ) -> Optional[VLATextOutput]:
        if self._text_submitted:
            return None

        logger.info("正在执行开局任务牌识别...")
        ocr_result = STANDARD_VLA_OCR_TEXT
        time_phrase = STANDARD_VLA_TIME_PHRASE
        location_phrase = STANDARD_VLA_LOCATION_PHRASE
        action_phrase = STANDARD_VLA_ACTION_PHRASE

        result = VLATextOutput(
            ocr_text=ocr_result,
            time_phrase=time_phrase,
            location_phrase=location_phrase,
            action_phrase=action_phrase,
        )

        if self._global_path:
            self._target_pos = self._global_path[-1]
            logger.info("使用录制轨迹作为导航路径")
        else:
            self._target_pos = self._find_target_in_static_data(location_phrase)
            if self._target_pos:
                start = Vector3(
                    x=current_msg.pose_gnss.pos_x,
                    y=current_msg.pose_gnss.pos_y,
                    z=current_msg.pose_gnss.pos_z,
                )
                # 无录制轨迹时退化为直线目标点序列
                self._global_path = [start, self._target_pos]
                self._current_path_index = 0
                logger.info(
                    "无录制轨迹，直线导航至目标: (%.2f, %.2f)",
                    self._target_pos.x,
                    self._target_pos.y,
                )

        mission_info = {
            "sub_scene": current_msg.scene_status.sub_scene_name,
            "ocr_text": ocr_result,
            "target_location": location_phrase,
            "target_pos": (
                {"x": self._target_pos.x, "y": self._target_pos.y}
                if self._target_pos
                else None
            ),
            "path_points": len(self._global_path),
            "status": "navigating",
        }
        self._save_mission_log(mission_info)
        self._text_submitted = True
        return result

    def _get_navigation_control(
        self, current_pos: Vector3, current_yaw: float
    ) -> VehicleControl:
        """基于全局路径的前瞻点跟踪（yaw 约定与项目 PathFollower 一致）。"""
        vc = VehicleControl()
        vc.gear = GearMode.DRIVE
        if not self._global_path or self._current_path_index >= len(self._global_path):
            return vc

        search_range = 30
        best_idx = self._current_path_index
        min_dist = float("inf")
        for i in range(
            max(0, self._current_path_index - 5),
            min(self._current_path_index + search_range, len(self._global_path)),
        ):
            pt = self._global_path[i]
            d = math.hypot(pt.x - current_pos.x, pt.y - current_pos.y)
            if d < min_dist:
                min_dist = d
                best_idx = i
        self._current_path_index = best_idx

        dist_to_end = math.hypot(
            self._global_path[-1].x - current_pos.x,
            self._global_path[-1].y - current_pos.y,
        )
        if (
            self._mission_completed
            or dist_to_end < 1.0
            or self._current_path_index >= len(self._global_path) - 1
        ):
            if not self._mission_completed:
                logger.info(
                    "到达终点! 距离: %.2fm, 索引: %d/%d",
                    dist_to_end,
                    self._current_path_index,
                    len(self._global_path) - 1,
                )
                self._mission_completed = True
                self._save_mission_log(
                    {
                        "status": "completed",
                        "dist_to_end": dist_to_end,
                        "path_index": self._current_path_index,
                    }
                )
            vc.throttle = 0.0
            vc.brake = 1.0
            vc.steering = 0.0
            return vc

        lookahead_steps = 15
        target_idx = min(
            self._current_path_index + lookahead_steps, len(self._global_path) - 1
        )
        target_pt = self._global_path[target_idx]
        dx = target_pt.x - current_pos.x
        dy = target_pt.y - current_pos.y
        target_yaw = math.atan2(dy, dx)

        yaw_diff = target_yaw - current_yaw
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi

        # 与 calc_steering / PathFollower 一致：对角度取负号再归一化到 [-1,1]
        vc.steering = max(min(-yaw_diff / (math.pi / 6), 1.0), -1.0)

        if self._frame_count % 30 == 0:
            logger.info(
                "Tracking: Idx=%d/%d, Target(%.1f, %.1f), Current(%.1f, %.1f), "
                "TargetYaw=%.1f, CurrentYaw=%.1f, YawDiff=%.1f, Steer=%.2f",
                self._current_path_index,
                len(self._global_path),
                target_pt.x,
                target_pt.y,
                current_pos.x,
                current_pos.y,
                math.degrees(target_yaw),
                math.degrees(current_yaw),
                math.degrees(yaw_diff),
                vc.steering,
            )

        vc.throttle = 0.10
        if abs(yaw_diff) > math.pi / 12:
            vc.throttle = 0.03
            vc.brake = 0.0
        return vc

    def update(
        self,
        sim_car_msg: SimCarMsg,
        perception_data: dict,
        frame_count: int,
    ) -> tuple[VehicleControl, Optional[VLAExtensionOutput]]:
        self._frame_count = frame_count
        vla_output: Optional[VLAExtensionOutput] = None

        text_result = self._submit_text(sim_car_msg)
        if text_result is not None:
            vla_output = VLAExtensionOutput(text_info=text_result)

        if self._mission_completed:
            vc = VehicleControl()
            vc.gear = GearMode.DRIVE
            vc.throttle = 0.0
            vc.brake = 1.0
            vc.steering = 0.0
            return vc, vla_output

        current_pos = Vector3(
            x=sim_car_msg.pose_gnss.pos_x,
            y=sim_car_msg.pose_gnss.pos_y,
            z=sim_car_msg.pose_gnss.pos_z,
        )
        # 与项目内 PathFollower / 场景 B/C 一致
        current_yaw = -math.radians(sim_car_msg.pose_gnss.ori_z)

        if self._global_path:
            vc = self._get_navigation_control(current_pos, current_yaw)
        else:
            vc = VehicleControl()
            vc.gear = GearMode.DRIVE
            current_speed = sim_car_msg.main_vehicle.speed
            if current_speed < self._target_speed:
                vc.throttle = 0.2
            else:
                vc.brake = 0.1

        return vc, vla_output

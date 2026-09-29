"""决策大脑

根据场景类型选择对应状态机，将感知数据转发给状态机，
输出车辆控制与 VLA 扩展信息。
"""

import logging
from typing import Optional

from metacar import (
    GearMode,
    SimCarMsg,
    VehicleControl,
    VLAExtensionOutput,
    SceneStaticData,
    RegionType,
)

from .state_machines.scene_a import SceneAStateMachine
from .state_machines.scene_b import SceneBStateMachine
from .state_machines.scene_c import SceneCStateMachine

logger = logging.getLogger(__name__)


class DecisionBrain:
    """决策大脑，负责场景识别与状态机调度

    :param static_data: 场景静态数据
    """

    def __init__(self, static_data: SceneStaticData) -> None:
        self._static_data = static_data
        self._state_machine = None
        self._scene_type: Optional[str] = None
        self._init_scene()

    def _init_scene(self):
        """根据静态数据判断场景类型并初始化对应状态机"""
        vla_ext = self._static_data.vla_extension
        if vla_ext is not None:
            has_function_zone = any(
                r.type == RegionType.FUNCTION_ZONE for r in vla_ext.regions
            )
            has_restricted = any(
                r.type == RegionType.RESTRICTED_PARKING for r in vla_ext.regions
            )
            if has_function_zone:
                self._scene_type = "C"
                self._state_machine = SceneCStateMachine(self._static_data)
                logger.info("场景类型: C（规则合规审查）")
            elif has_restricted:
                self._scene_type = "B"
                self._state_machine = SceneBStateMachine(self._static_data)
                logger.info("场景类型: B（禁停区违停检测）")
            else:
                # VLA 且无禁停/功能区 → 场景 A（任务牌导航停车）
                self._scene_type = "A"
                self._state_machine = SceneAStateMachine(self._static_data)
                logger.info("场景类型: A（任务牌 OCR + 轨迹导航）")
        else:
            self._scene_type = "A"
            self._state_machine = SceneAStateMachine(self._static_data)
            logger.info("场景类型: A（无 VLA 扩展，仍尝试轨迹导航）")

    @property
    def scene_type(self) -> Optional[str]:
        return self._scene_type

    def update(
        self,
        sim_car_msg: SimCarMsg,
        perception_data: dict,
        frame_count: int,
    ) -> tuple[VehicleControl, Optional[VLAExtensionOutput]]:
        """每帧更新决策

        :param sim_car_msg: 仿真动态信息
        :param perception_data: 感知数据，含 'ocr' 键
        :param frame_count: 当前帧计数
        :returns: (VehicleControl, VLAExtensionOutput | None)
        """
        if self._state_machine is not None:
            return self._state_machine.update(sim_car_msg, perception_data, frame_count)

        vc = VehicleControl()
        vc.gear = GearMode.DRIVE
        vc.throttle = 0.3
        return vc, None

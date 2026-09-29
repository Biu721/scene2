import logging
import sys
from pathlib import Path
import json
import os
import time

# 将项目根目录加入 sys.path，使 perception / decision 包可被导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import keyboard
import math
from metacar import SceneAPI, GearMode, VehicleControl, Vector3, SimCarMsg

from perception import PerceptionManager                                                                                                                                                                                    
from perception.sticker_classifier import StickerClassifier
from decision import DecisionBrain

# 是否使用 GUI 界面
USE_GUI = True

# 场景 B/C 感知间隔：每 N 帧跑一次 OCR
OCR_INTERVAL = 1

if USE_GUI:
    from gui import Dashboard

logging.basicConfig(
    filename="autodrive.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

current_gear = GearMode.DRIVE
use_keyboard = False

PATH_FILE_C = "path_C.json"
PATH_FILE_B = "path_B.json"
PATH_FILE_A = "recorded_trajectory.json"
PATH_FILE_LEGACY = "path_waypoints.json"


class PathRecorder:
    def __init__(self) -> None:
        self.recording = False
        self.points: list[dict] = []
        self._last_saved_ts = 0.0

    def toggle(self) -> bool:
        self.recording = not self.recording
        return self.recording

    def reset(self) -> None:
        self.points = []

    def add(self, sim_car_msg: SimCarMsg, frame_count: int) -> None:
        if not self.recording:
            return
        x = float(sim_car_msg.pose_gnss.pos_x)
        y = float(sim_car_msg.pose_gnss.pos_y)
        z = float(sim_car_msg.pose_gnss.pos_z)
        yaw_deg = float(sim_car_msg.pose_gnss.ori_z)
        speed = float(sim_car_msg.main_vehicle.speed)

        if self.points:
            last = self.points[-1]
            dx = x - float(last["x"])
            dy = y - float(last["y"])
            if (dx * dx + dy * dy) ** 0.5 < 0.6:
                return

        self.points.append({
            "frame": int(frame_count),
            "x": x,
            "y": y,
            "z": z,
            "yaw_deg": yaw_deg,
            "speed": speed,
        })

    def save(self, file_path: str) -> None:
        now = time.time()
        if now - self._last_saved_ts < 0.3:
            return
        self._last_saved_ts = now
        payload = {
            "version": 1,
            "points": self.points,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class PathFollower:
    def __init__(self, points: list[dict], base_speed: float = 1.2) -> None:
        self.points = points
        self._idx = 0
        self._last_nearest_frame = -999999
        self._initialized = False
        self.base_speed = float(base_speed)
        self.reached_end = False

    def _wrap_to_pi(self, a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def update(self, sim_car_msg: SimCarMsg, frame_count: int, current_gear: GearMode) -> VehicleControl:
        vc = VehicleControl()
        vc.gear = current_gear if current_gear in (GearMode.DRIVE, GearMode.REVERSE) else GearMode.DRIVE
        if not self.points:
            return vc

        x = float(sim_car_msg.pose_gnss.pos_x)
        y = float(sim_car_msg.pose_gnss.pos_y)
        yaw = -math.radians(float(sim_car_msg.pose_gnss.ori_z))
        speed = float(sim_car_msg.main_vehicle.speed)

        frame = int(frame_count)
        if not self._initialized:
            nearest_i = 0
            nearest_d2 = float("inf")
            for i in range(len(self.points)):
                tx0 = float(self.points[i]["x"])
                ty0 = float(self.points[i]["y"])
                dx0 = tx0 - x
                dy0 = ty0 - y
                d2 = dx0 * dx0 + dy0 * dy0
                if d2 < nearest_d2:
                    nearest_d2 = d2
                    nearest_i = i
            self._idx = nearest_i
            self._initialized = True
            self._last_nearest_frame = frame

        # 路径点较密/定位噪声下，阈值太小会导致 idx 卡住
        reach_radius = 2.2
        while self._idx < len(self.points) - 1:
            tx_check = float(self.points[self._idx]["x"])
            ty_check = float(self.points[self._idx]["y"])
            if math.hypot(tx_check - x, ty_check - y) <= reach_radius:
                self._idx += 1
            else:
                break

        tx = float(self.points[self._idx]["x"])
        ty = float(self.points[self._idx]["y"])
        dx = tx - x
        dy = ty - y
        angle_to_target = math.atan2(dy, dx)
        alpha = self._wrap_to_pi(angle_to_target - yaw)
        L = 2.7
        Ld = max(0.8, math.hypot(dx, dy))
        curvature = 2.0 * math.sin(alpha) / Ld
        steer_rad = math.atan(curvature * L)
        # 与项目内 calc_steering() 的符号约定保持一致（其内部对角度做了负号处理）
        vc.steering = max(min(math.degrees(-steer_rad) / 45.0, 1.0), -1.0)

        dist = math.hypot(dx, dy)
        target_speed = self.base_speed
        if dist < 2.0:
            target_speed = min(target_speed, 0.8)
        if self._idx >= len(self.points) - 2:
            target_speed = min(target_speed, 0.6)
            if dist < 0.5:
                self.reached_end = True

        diff = target_speed - speed
        if diff > 0.1:
            vc.throttle = min(diff * 0.4, 0.22)
            vc.brake = 0.0
        elif diff < -0.1:
            vc.throttle = 0.0
            vc.brake = min((-diff) * 0.6, 0.55)
        else:
            vc.throttle = 0.02
            vc.brake = 0.0

        return vc


def set_gear(gear: GearMode):
    global current_gear
    current_gear = gear


def toggle_keyboard():
    global use_keyboard
    use_keyboard = not use_keyboard


def get_vehicle_control_from_keyboard() -> VehicleControl:
    vc = VehicleControl()
    vc.gear = current_gear
    if keyboard.is_pressed("up") or keyboard.is_pressed("w"):
        value = 0.5 if keyboard.is_pressed("shift") else 1
        if current_gear == GearMode.DRIVE:
            vc.throttle = value
        elif current_gear == GearMode.REVERSE:
            vc.brake = value
    elif keyboard.is_pressed("down") or keyboard.is_pressed("s"):
        value = 0.5 if keyboard.is_pressed("shift") else 1
        if current_gear == GearMode.DRIVE:
            vc.brake = value
        elif current_gear == GearMode.REVERSE:
            vc.throttle = value
    if keyboard.is_pressed("left") or keyboard.is_pressed("a"):
        vc.steering = -1
    elif keyboard.is_pressed("right") or keyboard.is_pressed("d"):
        vc.steering = 1
    return vc


def calc_throttle_brake(
    current_speed: float, target_speed: float
) -> tuple[float, float]:
    K = 0.2
    B = 0.2
    acceleration = (target_speed - current_speed) * K + B
    if acceleration > 0:
        return min(acceleration, 1), 0
    return 0, min(-acceleration * 0.5, 1)


# Stanley 算法
def calc_steering(
    pos: Vector3, yaw: float, speed: float, trajectory: list[Vector3]
) -> float:
    K = 0.5
    for traj_pos in trajectory:
        if math.dist(traj_pos, pos) > K * speed:
            target_pos = traj_pos
            break
    else:
        target_pos = trajectory[-1]
    theta = (target_pos - pos).yaw_rad()
    steering_angle = (theta - yaw) % (2 * math.pi)
    if steering_angle > math.pi:
        steering_angle -= 2 * math.pi
    steering = math.degrees(-steering_angle) / 45
    return max(min(steering, 1), -1)


def get_vehicle_control_from_algorithm(msg: SimCarMsg) -> VehicleControl:
    vc = VehicleControl()
    vc.gear = GearMode.DRIVE
    # 目前只支持固定速度
    vc.throttle, vc.brake = calc_throttle_brake(msg.main_vehicle.speed, 8)
    vc.steering = calc_steering(
        Vector3(
            msg.pose_gnss.pos_x,
            msg.pose_gnss.pos_y,
            msg.pose_gnss.pos_z,
        ),
        -math.radians(msg.pose_gnss.ori_z),
        msg.main_vehicle.speed,
        msg.trajectory,
    )
    return vc


def _get_left_camera_frames(frames: list) -> list:
    """筛选左相机帧（通常 camera_id 包含 'left' 或为第一个相机）"""
    left_frames = [f for f in frames if "left" in f.id.lower()]
    if not left_frames:
        # 回退：取第一个相机
        left_frames = frames[:1] if frames else []
    return left_frames


def main():
    api = SceneAPI()

    recorder = PathRecorder()
    scene_path_file = PATH_FILE_LEGACY
    path_file_path = str(PROJECT_ROOT / scene_path_file)
    path_follower: PathFollower | None = None
    enable_path_follow = False
    path_follow_base_speed = 1.2

    print("正在连接仿真器...", flush=True)
    api.connect()
    static_data = api.get_scene_static_data()
    print("仿真器连接成功，获取到静态数据", flush=True)

    # 判断是否为 VLA 场景
    is_vla = static_data.vla_extension is not None
    print(f"场景类型: {'VLA' if is_vla else '普通'}, 扩展信息是否存在: {is_vla}", flush=True)

    # 初始化感知模块
    perception_manager = PerceptionManager(use_real_ocr=True)
    sticker_classifier = StickerClassifier()
    print("感知模块初始化完成", flush=True)

    # 初始化决策模块
    decision_brain = DecisionBrain(static_data) if is_vla else None
    if decision_brain:
        print(f"决策大脑初始化完成，当前场景类型: {decision_brain.scene_type}", flush=True)

    # 根据场景类型选择路径文件。
    # 约定：A 用 recorded_trajectory.json（由 SceneAStateMachine 加载跟线）；
    #      B 用 path_B.json；C 用 path_C.json。
    scene_type = decision_brain.scene_type if decision_brain else "A"
    if scene_type == "C":
        scene_path_file = PATH_FILE_C
        enable_path_follow = True
        path_follow_base_speed = 1.2
    elif scene_type == "B":
        scene_path_file = PATH_FILE_B
        enable_path_follow = True
        path_follow_base_speed = 2.0
    elif scene_type == "A":
        scene_path_file = PATH_FILE_A
        # 场景 A 导航由 SceneAStateMachine 内部完成，不走 PathFollower
        enable_path_follow = False
        path_follow_base_speed = 1.5
        traj_path = PROJECT_ROOT / PATH_FILE_A
        if traj_path.exists():
            print(f"[SceneA] 将使用录制轨迹: {traj_path}", flush=True)
        else:
            print(f"[SceneA] 未找到 {PATH_FILE_A}，将尝试静态数据直线导航", flush=True)
    else:
        scene_path_file = PATH_FILE_LEGACY
        enable_path_follow = False
    path_file_path = str(PROJECT_ROOT / scene_path_file)

    def load_path_follower() -> PathFollower | None:
        # Scene C：若新文件不存在但 legacy 存在，则读取 legacy 并迁移保存为 path_C.json
        candidate_path = path_file_path
        legacy_path = str(PROJECT_ROOT / PATH_FILE_LEGACY)
        try_migrate = (scene_path_file == PATH_FILE_C) and (not os.path.exists(candidate_path)) and os.path.exists(legacy_path)
        if not os.path.exists(candidate_path):
            if try_migrate:
                candidate_path = legacy_path
            else:
                return None

        try:
            with open(candidate_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            pts = payload.get("points", []) if isinstance(payload, dict) else []
            if isinstance(pts, list) and pts:
                if try_migrate:
                    try:
                        with open(path_file_path, "w", encoding="utf-8") as wf:
                            json.dump(payload, wf, ensure_ascii=False, indent=2)
                        print(f"[Path] 已将 {PATH_FILE_LEGACY} 迁移为 {scene_path_file}", flush=True)
                    except Exception:
                        logger.exception("迁移路径文件失败")

                # 抽稀：点太密会导致逐点模式抖动、难以收敛
                min_keep_dist = 1.5
                ds: list[dict] = []
                last_x = None
                last_y = None
                for p in pts:
                    try:
                        px = float(p.get("x"))
                        py = float(p.get("y"))
                    except Exception:
                        continue
                    if last_x is None:
                        ds.append(p)
                        last_x, last_y = px, py
                        continue
                    if math.hypot(px - last_x, py - last_y) >= min_keep_dist:
                        ds.append(p)
                        last_x, last_y = px, py
                if len(ds) < 2:
                    ds = pts
                print(
                    f"已加载路径文件: {candidate_path}, points={len(pts)} -> downsample={len(ds)} (min_dist={min_keep_dist})",
                    flush=True,
                )
                logger.info("PathFollower 已启用: %s downsample=%d", candidate_path, len(ds))
                return PathFollower(ds, base_speed=path_follow_base_speed)
        except Exception:
            logger.exception("加载路径文件失败")
        return None

    path_follower = load_path_follower() if enable_path_follow else None
    if enable_path_follow and path_follower is None:
        print(f"[Path] 未找到路径文件: {path_file_path}，将回退算法控制（不会走录制路径）", flush=True)
        logger.warning("未找到路径文件，回退状态机控车: %s", path_file_path)
    elif path_follower is not None:
        print(f"[Path] 自动跟线已开启: {scene_path_file}", flush=True)

    keyboard.add_hotkey("space", api.retry_level)
    keyboard.add_hotkey("n", api.skip_level)

    keyboard.add_hotkey("r", lambda: set_gear(GearMode.REVERSE))
    keyboard.add_hotkey("f", lambda: set_gear(GearMode.DRIVE))
    keyboard.add_hotkey("t", lambda: set_gear(GearMode.NEUTRAL))
    keyboard.add_hotkey("g", lambda: set_gear(GearMode.PARKING))

    def on_press_c():
        global use_keyboard
        use_keyboard = not use_keyboard
        mode = "键盘模式" if use_keyboard else "自动模式"
        follower_state = "ON" if (path_follower is not None and enable_path_follow) else "OFF"
        print(
            f"\n[Mode] 切换为{mode} (Scene={scene_type}, PathFollower={follower_state}, file={scene_path_file})",
            flush=True,
        )

    keyboard.add_hotkey("c", on_press_c)

    def on_press_p():
        nonlocal path_follower
        is_on = recorder.toggle()
        if is_on:
            recorder.reset()
            print(f"\n[Path] 开始录制路径: {scene_path_file}", flush=True)
        else:
            try:
                recorder.save(path_file_path)
                print(f"\n[Path] 停止录制并保存: {scene_path_file}, points={len(recorder.points)}", flush=True)
                if enable_path_follow:
                    path_follower = load_path_follower()
                    follower_state = "ON" if path_follower is not None else "OFF"
                    print(f"[Path] PathFollower reload: {follower_state}", flush=True)
            except Exception:
                logger.exception("保存路径失败")

    keyboard.add_hotkey("p", on_press_p)

    if USE_GUI:
        dashboard = Dashboard(static_data)
        logger.info("启动 GUI 界面")
        print("GUI 界面启动", flush=True)
    
    logger.info("开始场景 (VLA=%s, 场景类型=%s)", is_vla, decision_brain.scene_type if decision_brain else "A")
    print("进入主循环...", flush=True)

    def on_press_s():
        if decision_brain and hasattr(decision_brain, "_state_machine"):
            sm = decision_brain._state_machine
            if hasattr(sm, "force_report"):
                # 记录手动上报前的状态
                old_state = getattr(sm, "_state", "Unknown")
                if sm.force_report():
                    new_state = getattr(sm, "_state", "Unknown")
                    print(f"\n[Manual] 手动触发上报: {old_state} -> {new_state}。请等待结算...", flush=True)
                else:
                    print(f"\n[Manual] 无法触发上报 (当前状态: {old_state})，请检查是否已在 DONE 状态。", flush=True)

    keyboard.add_hotkey("enter", on_press_s)
    print("已添加手动上报热键: [Enter]", flush=True)

    frame_count = 0
    try:
        for sim_car_msg, frames in api.main_loop():
            if frame_count == 0:
                print("成功进入 main_loop，收到第一帧数据", flush=True)
            frame_count += 1
            
            # 记录关键状态，排查提前停止原因
            status = sim_car_msg.scene_status
            if frame_count % 50 == 0:
                logger.info("运行中: 帧=%d, 已用时间=%.1fs, 限制=%.1fs, 子场景=%s", 
                            frame_count, status.used_time, status.time_limit, status.sub_scene_name)

            vla_extension = None
            if is_vla and decision_brain is not None:
                left_frames = _get_left_camera_frames(frames)
                perception_data: dict = {}

                # 场景 B/C 需要 OCR；场景 A 用标准牌面文本，无需每帧 OCR
                if scene_type in ("B", "C"):
                    if frame_count % OCR_INTERVAL == 0 and left_frames:
                        ocr_result = perception_manager.process(
                            left_frames, timestamp=frame_count
                        )
                        perception_data.update(ocr_result)
                        perception_data["frames"] = left_frames
                        perception_data["sticker_classifier"] = sticker_classifier

                # 决策（即使在键盘模式下也运行，以更新状态机和获取上报/文本提交）
                algo_control, vla_extension = decision_brain.update(
                    sim_car_msg, perception_data, frame_count
                )

                if use_keyboard:
                    vehicle_control = get_vehicle_control_from_keyboard()
                    recorder.add(sim_car_msg, frame_count)
                    # 键盘模式下覆盖算法控制，但保留 vla_extension 用于上报/文本提交
                elif scene_type == "A":
                    # 场景 A：由 SceneAStateMachine 负责文本提交 + 轨迹导航停车
                    vehicle_control = algo_control
                    if frame_count % 50 == 0:
                        sm = getattr(decision_brain, "_state_machine", None)
                        done = bool(getattr(sm, "mission_completed", False))
                        print(
                            f"[SceneA] frame={frame_count} done={done} "
                            f"steer={vehicle_control.steering:.2f} "
                            f"thr={vehicle_control.throttle:.2f} brk={vehicle_control.brake:.2f}",
                            flush=True,
                        )
                elif path_follower is not None:
                    vehicle_control = path_follower.update(
                        sim_car_msg, frame_count, current_gear
                    )
                    # 自动触发 Enter 逻辑（B/C 上报）
                    if path_follower.reached_end:
                        print("\n[Path] 已到达路径终点，自动触发上报...", flush=True)
                        on_press_s()
                        path_follower.reached_end = False  # 防止重复触发
                    if frame_count % 50 == 0:
                        dx = float(path_follower.points[path_follower._idx]["x"]) - float(
                            sim_car_msg.pose_gnss.pos_x
                        )
                        dy = float(path_follower.points[path_follower._idx]["y"]) - float(
                            sim_car_msg.pose_gnss.pos_y
                        )
                        print(
                            f"[PathFollow] frame={frame_count} idx={path_follower._idx}/{len(path_follower.points)} "
                            f"dist={math.hypot(dx, dy):.2f} steer={vehicle_control.steering:.2f} "
                            f"thr={vehicle_control.throttle:.2f} brk={vehicle_control.brake:.2f}",
                            flush=True,
                        )
                else:
                    vehicle_control = algo_control
            else:
                if use_keyboard:
                    vehicle_control = get_vehicle_control_from_keyboard()
                    recorder.add(sim_car_msg, frame_count)
                else:
                    if path_follower is not None:
                        vehicle_control = path_follower.update(sim_car_msg, frame_count, current_gear)
                        # 自动触发 Enter 逻辑
                        if path_follower.reached_end:
                            print("\n[Path] 已到达路径终点，自动触发上报...", flush=True)
                            on_press_s()
                            path_follower.reached_end = False  # 防止重复触发
                        if frame_count % 50 == 0:
                            dx = float(path_follower.points[path_follower._idx]["x"]) - float(sim_car_msg.pose_gnss.pos_x)
                            dy = float(path_follower.points[path_follower._idx]["y"]) - float(sim_car_msg.pose_gnss.pos_y)
                            print(
                                f"[PathFollow] frame={frame_count} idx={path_follower._idx}/{len(path_follower.points)} "
                                f"dist={math.hypot(dx, dy):.2f} steer={vehicle_control.steering:.2f} "
                                f"thr={vehicle_control.throttle:.2f} brk={vehicle_control.brake:.2f}",
                                flush=True,
                            )
                    else:
                        vehicle_control = get_vehicle_control_from_algorithm(sim_car_msg)
                        if scene_type == "B":
                            # 场景 B 没有写死路径时，限制算法自动速度，避免过快
                            vehicle_control.throttle, vehicle_control.brake = calc_throttle_brake(
                                sim_car_msg.main_vehicle.speed, 6
                            )

            api.set_vehicle_control(vehicle_control, vla_extension=vla_extension)
            if USE_GUI:
                dashboard.update(sim_car_msg)

        print("main_loop 已结束（仿真器不再发送帧）", flush=True)

    except Exception as e:
        logger.exception("主循环发生异常")
    finally:
        logger.info("结束场景")
        if USE_GUI:
            dashboard.quit()
        logger.info("关闭 GUI 界面")


if __name__ == "__main__":
    main()

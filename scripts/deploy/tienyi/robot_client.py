#!/usr/bin/env python3
"""TienYi dual-arm remote inference client (blocking chunk execution).

Adapted from the TienKung client (scripts/deploy/tienkung/robot_client.py) for the
pi05_tienyi_eggplant_delta checkpoints. Runs on the robot host with the robot's
Python environment (xRocs / httpx / cv2), NOT this repo's venv.

The whole chain uses the TRAINING layout (see docs/推理部署方案_tienyi.md):

    0:7   left arm joints (rad)
    7:14  right arm joints (rad)
    14    left gripper opening [0, 1]
    15    right gripper opening [0, 1]

Note: unlike the TienKung lint-roller task (left arm idle), the eggplant task moves
BOTH arms, so there is no "left arm stays still" expectation here.

Camera note: the training dataset overhead view is `camera_top` (renamed from the
platform's `camera_head`). This client fuzzy-matches "top" OR "head" for the overhead
camera, so it works whether the robot exposes it as top or head.
"""

import argparse
import base64
import contextlib
import ctypes
import dataclasses
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.parse

import cv2
import httpx
import numpy as np

try:
    import tomllib as tomli  # py3.11+
except ImportError:  # pragma: no cover
    import tomli  # type: ignore


def _bootstrap_ros2_python_paths() -> None:
    """Make ROS2 generated Python message packages importable."""
    ws_root = Path(os.environ.get("ROS2_WS_PATH", "/home/ubuntu/ros2ws"))
    install_dir = ws_root / "install"
    if not install_dir.exists():
        return

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [install_dir / f"lib/{py_ver}/site-packages"]
    for pkg_dir in install_dir.iterdir():
        candidates.append(pkg_dir / f"local/lib/{py_ver}/dist-packages")
        candidates.append(pkg_dir / f"lib/{py_ver}/site-packages")

    for p in candidates:
        if p.exists():
            hric_init = p / "hric_msgs/msg/__init__.py"
            if hric_init.exists() and "FloatBaseRPYZCommandFeedback" not in hric_init.read_text():
                # Older workspace hric_msgs shadows the system package required by xRocs.
                continue
            p_str = str(p)
            if p_str not in sys.path:
                sys.path.insert(0, p_str)

    lib_candidates = [install_dir / "lib"]
    for pkg_dir in install_dir.iterdir():
        lib_candidates.append(pkg_dir / "lib")
        lib_candidates.append(pkg_dir / "local/lib")

    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [x for x in existing_ld.split(":") if x]
    for p in reversed(lib_candidates):
        if p.exists():
            p_str = str(p)
            if p_str not in ld_parts:
                ld_parts.insert(0, p_str)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)


def _preload_workspace_type_support_libs() -> None:
    """Preload workspace ROS2 message libs to avoid ABI mismatch."""
    ws_root = Path(os.environ.get("ROS2_WS_PATH", "/home/ubuntu/ros2ws"))
    lib_dir = ws_root / "install/bodyctrl_msgs/lib"
    if not lib_dir.exists():
        return

    preload_order = [
        lib_dir / "libbodyctrl_msgs__rosidl_generator_c.so",
        lib_dir / "libbodyctrl_msgs__rosidl_typesupport_c.so",
        lib_dir / "libbodyctrl_msgs__rosidl_generator_py.so",
    ]
    for so_path in preload_order:
        if so_path.exists():
            with contextlib.suppress(OSError):
                ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)


_bootstrap_ros2_python_paths()
_preload_workspace_type_support_libs()

# Make local xRocs source importable when package is not installed.
if "xrocs" not in sys.modules:
    for candidate in (
        Path("/home/ubuntu/xRocs"),
        Path(__file__).resolve().parent / "xRocs",
        Path.cwd() / "xRocs",
    ):
        if (candidate / "xrocs" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break

from xrocs.common.data_type.joints_data import Joints  # noqa: E402
from xrocs.core.config_loader import ConfigLoader  # noqa: E402
from xrocs.core.station_loader import StationLoader  # noqa: E402
from xrocs.entity.camera.camera_loader import CameraLoader  # noqa: E402
from xrocs.entity.hand.hand_loader import HandLoader  # noqa: E402
from xrocs.utils.logger.logger_loader import logger  # noqa: E402

DEFAULT_CONFIG_PATH = "/home/ubuntu/Documents/configuration.toml"
INFER_URL = "http://127.0.0.1:18000/inference"
DEFAULT_TASK = "Place the eggplant on the plate"  # 必须与训练 prompt 逐字一致
CAMERA_WARMUP_SECS = 8.0
CONTROL_DT = 1 / 29  # 数据集 fps=29, 与训练时间步一致
MAX_DELTA_PER_STEP = 0.08
PRINT_EVERY = 5

ARM_DOF = 7
ACTION_DIM = 16

# Overhead camera key aliases: the dataset uses camera_top (renamed from the platform's
# camera_head), so accept either "top" or "head" for the overhead view.
CAMERA_ALIASES: dict[str, tuple[str, ...]] = {
    "top": ("top", "head"),
    "left": ("left",),
    "right": ("right",),
}

# eggplant 数据集 554 条 episode 首帧位姿均值(训练布局 [L7, R7, Lg, Rg])
COLLECTION_INIT_POSE = np.asarray(
    [
        # 左臂 0:7
        0.122871, 0.116640, -0.129596, -1.853367, -1.278654, -0.054483, -0.000901,
        # 右臂 7:14
        0.145657, -0.131415, 0.346642, -1.964025, 1.511000, 0.265127, 0.115378,
        # 左夹爪 14, 右夹爪 15
        0.011765, 0.011765,
    ],
    dtype=np.float64,
)

CHUNK_CONFIRM = True  # 首跑逐块确认; 跑顺后可改 False
CHUNK_EXEC_STEPS = 15

LEFT_ARM_LIMITS = [
    (math.radians(-170), math.radians(170)),
    (math.radians(-15), math.radians(150)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-45), math.radians(60)),
    (math.radians(-95), math.radians(75)),
]
RIGHT_ARM_LIMITS = [
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-45), math.radians(60)),
    (math.radians(-75), math.radians(95)),
]


@dataclasses.dataclass(frozen=True)
class InferenceResponse:
    actions: np.ndarray
    rtt_ms: float
    request_id: str | None
    policy_timing: dict
    server_timing: dict


class TienyiInferenceClient:
    def __init__(self, config_path: str, task_name: str, infer_url: str = INFER_URL, *, dry_run: bool = False):
        self.config_path = config_path
        self.current_task = task_name
        self.infer_url = infer_url
        self.dry_run = dry_run
        self.image_color_order = "RGB"
        self.camera_short_keys = ("top", "left", "right")

        with open(config_path, "rb") as f:
            self.cfg = tomli.load(f)

        cfg_loader = ConfigLoader(config_path)
        cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()

        robot_handle = self.robot_station.get_robot_handle()["robot"]
        self.arm_ctrler = robot_handle.dual_arm_ctrler

        hand_loader = HandLoader()
        self.hands = hand_loader.instantiate_hands(self.cfg.get("hand", {}))
        for name, hand in self.hands.items():
            try:
                hand.connect()
                logger.success(f"hand connected: {name}")
            except Exception as exc:
                logger.warning(f"hand connect failed {name}: {exc}")

        camera_loader = CameraLoader()
        self.cameras = camera_loader.instantiate_cameras(self.cfg.get("camera", {}))
        self.camera_keys = tuple(self.cameras.keys())

        self.http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            headers={
                "User-Agent": f"TienyiClient/{urllib.parse.quote(task_name)}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )

        print(f"HTTP推理地址: {self.infer_url}")
        print(f"相机: {list(self.camera_keys)}")
        if self.dry_run:
            print("*** DRY-RUN 模式: 不向机器人下发任何指令, 仅打印 ***")

    @staticmethod
    def _extract_scalar(x: Any) -> float:
        if x is None:
            return 0.0
        if isinstance(x, int | float):
            return float(x)
        if hasattr(x, "get_radian_ndarray"):
            arr = x.get_radian_ndarray()
            if len(arr) > 0:
                return float(arr[0])
            return 0.0
        if hasattr(x, "__len__") and not isinstance(x, str | bytes):
            try:
                return float(x[0]) if len(x) > 0 else 0.0
            except Exception:
                return 0.0
        return 0.0

    def _get_hand_state(self) -> tuple[float, float]:
        left_pos, right_pos = 0.0, 0.0
        if "left" in self.hands:
            try:
                left_pos = self._extract_scalar(self.hands["left"].get_current_joint())
            except Exception:
                left_pos = 0.0
        if "right" in self.hands:
            try:
                right_pos = self._extract_scalar(self.hands["right"].get_current_joint())
            except Exception:
                right_pos = 0.0
        return float(np.clip(left_pos, 0.0, 1.0)), float(np.clip(right_pos, 0.0, 1.0))

    def _get_arm_state(self) -> np.ndarray:
        joints = self.arm_ctrler.get_current_joint().get_radian_ndarray()
        return np.asarray(joints, dtype=np.float32).reshape(-1)[:14]

    def _get_state_for_server(self) -> np.ndarray:
        """State in the TRAINING layout: [left arm 7, right arm 7, left grip, right grip]."""
        arm = self._get_arm_state()
        if len(arm) < 14:
            arm = np.pad(arm, (0, 14 - len(arm)))
        left_hand, right_hand = self._get_hand_state()
        return np.asarray(np.concatenate([arm[:14], [left_hand], [right_hand]]), dtype=np.float32)

    def _read_cameras(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for name, camera in self.cameras.items():
            rgb, _ = camera.read()
            if isinstance(rgb, np.ndarray) and rgb.size > 0:
                images[name] = rgb
        return images

    def _obs_has_any_frame(self) -> bool:
        return len(self._read_cameras()) > 0

    def _warmup_camera(self) -> bool:
        start = time.time()
        while time.time() - start < CAMERA_WARMUP_SECS:
            if self._obs_has_any_frame():
                return True
            time.sleep(0.05)
        return False

    def _wait_for_images(self, timeout: float = 5.0) -> dict[str, np.ndarray]:
        deadline = time.monotonic() + timeout
        while True:
            images = self._read_cameras()
            if images:
                return images
            if time.monotonic() > deadline:
                raise RuntimeError(f"相机在 {timeout:.1f}s 内未返回有效帧, 请检查连接")
            time.sleep(0.02)

    def _encode_b64_jpg(self, img: np.ndarray, quality: int = 90) -> str:
        if img.ndim == 1 and img.dtype == np.uint8:
            decoded = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Unexpected encoded image buffer shape: {img.shape}")
            return base64.b64encode(img.tobytes()).decode("utf-8")

        if img.dtype != np.uint8:
            x = img.astype(np.float32)
            if x.size and float(np.nanmax(x)) <= 1.0 + 1e-6:
                x = x * 255.0
            img = np.clip(x, 0, 255).astype(np.uint8)

        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        bgr = img[:, :, ::-1] if self.image_color_order == "RGB" else img
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _pick_camera(self, images: dict[str, np.ndarray], short_key: str) -> np.ndarray | None:
        aliases = CAMERA_ALIASES.get(short_key, (short_key,))
        # Exact match first (across all aliases), then fuzzy substring match.
        for alias in aliases:
            for key, frame in images.items():
                if key.lower() == alias:
                    return frame
        for alias in aliases:
            for key, frame in images.items():
                if alias in key.lower():
                    return frame
        return None

    def _extract_images_for_server(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        images_b64 = {}
        for short_key in self.camera_short_keys:
            frame = self._pick_camera(images, short_key)
            if frame is None:
                raise KeyError(f"Missing camera '{short_key}', available={list(images.keys())}")
            images_b64[short_key] = self._encode_b64_jpg(frame)
        return images_b64

    def _build_payload(self, images: dict[str, np.ndarray], request_id: str | None) -> dict:
        state = self._get_state_for_server()
        return {
            "images": self._extract_images_for_server(images),
            "state": state.tolist(),
            "task": self.current_task,
            "request_id": request_id,
        }

    def request_action_chunk(
        self,
        images: dict[str, np.ndarray],
        *,
        request_id: str | None = None,
        timeout_s: float | None = None,
    ) -> InferenceResponse:
        payload = self._build_payload(images, request_id)
        headers = {
            "X-Task-Name": urllib.parse.quote(self.current_task),
            "Content-Type": "application/json; charset=utf-8",
        }

        t0 = time.monotonic()
        resp = self.http_client.post(self.infer_url, json=payload, headers=headers, timeout=timeout_s)
        rtt_ms = (time.monotonic() - t0) * 1000

        if resp.status_code >= 400:
            print("[HTTP ERROR]", resp.status_code, resp.text[:500])
            resp.raise_for_status()

        data = resp.json()
        if data.get("status") != "success" or "action_pred" not in data:
            raise RuntimeError(f"Bad response: {data}")

        actions = np.asarray(data["action_pred"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[1] < ACTION_DIM:
            raise ValueError(f"action_pred dim < {ACTION_DIM}: {actions.shape}")

        return InferenceResponse(
            actions=actions[:, :ACTION_DIM],
            rtt_ms=rtt_ms,
            request_id=data.get("request_id", request_id),
            policy_timing=data.get("policy_timing", {}),
            server_timing=data.get("server_timing", {}),
        )

    def _send_one_hand(self, hand_name: str, close_value: float) -> None:
        if hand_name not in self.hands:
            return
        hand = self.hands[hand_name]
        value = float(np.clip(close_value, 0.0, 1.0))
        try:
            hand.set_target_joint(Joints(np.array([value], dtype=np.float64), num_of_dofs=1))
            return
        except Exception:
            pass
        try:
            hand.set_target_joint(np.array([value], dtype=np.float64))
            return
        except Exception:
            pass
        try:
            if value < 0.5 and hasattr(hand, "open"):
                hand.open()
            elif value >= 0.5 and hasattr(hand, "close"):
                hand.close()
        except Exception as exc:
            logger.warning(f"hand cmd failed ({hand_name}): {exc}")

    def _send_hands(self, left_pos: float, right_pos: float) -> None:
        self._send_one_hand("left", left_pos)
        self._send_one_hand("right", right_pos)

    @staticmethod
    def _slew_limit(q_prev: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        dq = np.clip(q_target - q_prev, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
        return q_prev + dq

    @staticmethod
    def _clip_action(action: np.ndarray) -> np.ndarray:
        """Training layout: left arm 0:7, right arm 7:14, grippers 14/15."""
        action = np.asarray(action, dtype=np.float32).copy()[:ACTION_DIM]
        for i, (lo, hi) in enumerate(LEFT_ARM_LIMITS):
            action[i] = np.clip(action[i], lo, hi)
        for i, (lo, hi) in enumerate(RIGHT_ARM_LIMITS):
            action[7 + i] = np.clip(action[7 + i], lo, hi)
        action[14] = np.clip(action[14], 0.0, 1.0)
        action[15] = np.clip(action[15], 0.0, 1.0)
        return action

    def _command_action(self, q_abs: np.ndarray, q_prev: np.ndarray, step_label: str = "") -> np.ndarray:
        q_abs = self._clip_action(q_abs)
        q_cmd = self._slew_limit(q_prev, q_abs)
        # Training layout: q_cmd[:14] is already [left arm 7, right arm 7].
        arm_target = q_cmd[:14].astype(np.float64)
        if self.dry_run:
            print(f"[DRY-RUN] arm_target={np.round(arm_target, 4).tolist()} "
                  f"Lg={q_cmd[14]:.3f} Rg={q_cmd[15]:.3f}")
        else:
            self.arm_ctrler.set_cmd_pos(arm_target, timeout=0.0)
            self._send_hands(float(q_cmd[14]), float(q_cmd[15]))
        if step_label:
            print(
                f"[执行] {step_label} "
                f"L0={q_cmd[0]:.3f} R0={q_cmd[7]:.3f} Lg={q_cmd[14]:.3f} Rg={q_cmd[15]:.3f}"
            )
        return q_cmd

    def _execute_actions(self, actions: np.ndarray, q_prev: np.ndarray, label: str) -> tuple[np.ndarray, int]:
        executed = 0
        for idx, action in enumerate(actions):
            t_step_start = time.monotonic()
            try:
                step_label = f"{label} step={idx + 1}/{len(actions)}" if (idx + 1) % PRINT_EVERY == 0 else ""
                q_prev = self._command_action(action, q_prev, step_label=step_label)
            except Exception as exc:
                print(f"[执行] command failed at {label} step {idx}: {exc}")
                break
            executed += 1
            elapsed = time.monotonic() - t_step_start
            remaining = CONTROL_DT - elapsed
            if remaining > 0:
                time.sleep(remaining)
        return q_prev, executed

    @staticmethod
    def _format_state(q: np.ndarray) -> str:
        return (
            f"L_arm={np.round(q[0:7], 4).tolist()} "
            f"R_arm={np.round(q[7:14], 4).tolist()} "
            f"Lg={q[14]:.4f} Rg={q[15]:.4f}"
        )

    def prepare(self):
        print("移动到 eggplant 数据集首帧均值位姿。")
        # Training layout: COLLECTION_INIT_POSE[:14] is [left arm 7, right arm 7].
        target_arm = COLLECTION_INIT_POSE[:14].astype(np.float64)
        if self.dry_run:
            print(f"[DRY-RUN] prepare arm_target={np.round(target_arm, 4).tolist()} "
                  f"Lg={COLLECTION_INIT_POSE[14]:.3f} Rg={COLLECTION_INIT_POSE[15]:.3f}")
        else:
            self.arm_ctrler.reach_target_joint(Joints(target_arm, num_of_dofs=14))
            self._send_hands(float(COLLECTION_INIT_POSE[14]), float(COLLECTION_INIT_POSE[15]))
            time.sleep(1.0)

        # 与外层控制周期对齐, 降低底层伺服与上层循环的时间步失配风险。
        # dry-run 语义为"不向机器人下发任何指令", 控制器参数设置同样跳过。
        for method_name, value in (
            ("set_dt", float(CONTROL_DT)),
            ("set_gain", 150),
            ("set_lookahead_time", 0.1),
        ):
            if self.dry_run:
                print(f"[DRY-RUN] 跳过控制器参数设置: {method_name}({value})")
                continue
            method = getattr(self.arm_ctrler, method_name, None)
            if callable(method):
                try:
                    method(value)
                    print(f"控制器参数已设置: {method_name}({value})")
                except Exception as exc:
                    print(f"控制器参数设置失败 {method_name}: {exc}")

        # 核对当前读数与数据集初始位姿(同时验证真机关节顺序与数据集一致)。
        current = self._get_state_for_server()
        diff = np.abs(current - COLLECTION_INIT_POSE.astype(np.float32))
        print(f"[核对] 当前 state: {self._format_state(current)}")
        print(f"[核对] 与 INIT_POSE 最大偏差: {diff.max():.4f} rad (dim {int(diff.argmax())})")
        logger.success("Tienyi dual-arm prepare done.")

    def run(self):
        print("开始远程推理循环(blocking 模式)...")
        if not self._warmup_camera():
            print(f"相机在 {CAMERA_WARMUP_SECS:.1f}s 内仍未收到帧, 但仍会继续尝试。")

        try:
            while True:
                images = self._wait_for_images()
                response = self.request_action_chunk(images)
                action_chunk = response.actions
                steps_to_run = min(CHUNK_EXEC_STEPS, len(action_chunk))
                q_start = self._get_state_for_server()
                print(
                    f"[推理返回] action_chunk={action_chunk.shape} rtt_ms={response.rtt_ms:.1f}"
                )
                print(f"[推理返回] 当前 state: {self._format_state(q_start)}")
                if len(action_chunk):
                    print(f"[推理返回] 首步 action: {self._format_state(action_chunk[0])}")
                if CHUNK_CONFIRM:
                    answer = input(
                        f"[安全开关] chunk_len={len(action_chunk)}, will_run={steps_to_run}. 按回车执行; 输入 q 退出: "
                    ).strip().lower()
                    if answer == "q":
                        return
                self._execute_actions(action_chunk[:steps_to_run], q_start, label="blocking")
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.http_client.close()
            print("HTTP客户端连接已关闭")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TienYi dual-arm remote inference client")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="机器人配置文件路径")
    parser.add_argument("--infer-url", default=INFER_URL, help="远程推理服务地址")
    parser.add_argument("--task", default=DEFAULT_TASK, help="任务 prompt(必须与训练逐字一致)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="不向机器人下发任何指令(含运动目标与控制器参数设置), 仅打印(真机前核验用)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"任务: {args.task}")
    print("正在初始化天翼双臂控制...")
    robot = TienyiInferenceClient(
        config_path=args.config, task_name=args.task, infer_url=args.infer_url, dry_run=args.dry_run
    )
    robot.prepare()
    print("\n" + "=" * 60)
    print("开始远程推理循环(天翼双臂7+7轴, 训练布局 [L7,R7,Lg,Rg])")
    print("本地只负责控制, 推理在远程服务器进行")
    print("=" * 60 + "\n")
    robot.run()


if __name__ == "__main__":
    main()

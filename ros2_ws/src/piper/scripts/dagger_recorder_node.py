#!/usr/bin/env python3
"""DAgger session orchestrator + recorder (Form C: dual-dataset).

Form C — records BOTH policy rollouts AND human corrections to separate
datasets, binary-compatible with upstream kai0_dagger (HDF5 ≠ parquet but
schema-aligned). This is the open-source RECAP / AWBC advantage pipeline
prerequisite (see docs/deployment/strategy/awbc_implementation_plan.md).

Datasets (version dir from KAI0_DATASET_VERSION, current = v4):
  <task>/inference/<vN>/<date-vN>/    ← policy rollouts (intervention=0)
  <task>/dagger/<vN>/<date-vN>/       ← human corrections (intervention=1)

Lifecycle (4 states; pedal does NOT change state):

  POLICY_RUN   : policy publishes /master/joint_* → slave follows.
                 *Records to inference dataset, intervention=0.*
                 ↓ ANY freedrive switch rising edge (after slave-moved grace)
  ALIGNING     : (1) halt policy + finalize inference episode
                 (2) master into drag mode (encoder publish, slave follows)
                 ↓ both freedrive switches ON (handled inside _do_takeover)
  HUMAN_RECORD : drag mode active for as long as both switches stay ON.
                 Pedal toggles a SEPARATE _recording flag that drives the
                 dagger writer open/close — state stays HUMAN_RECORD.
                 *Records to dagger dataset (intervention=1) WHEN _recording.*
                 ↓ any switch falling
  RETURNING    : (1) close dagger writer if open
                 (2) re-enable masters (EnableArm + CAN_CTRL)
                 (3) /policy/execute=true → policy resumes
                 (4) open new inference episode
                 ↓ done
  POLICY_RUN   : back to start

Pedal toggle (KAI0 official Space ↔ s key equivalent):
  - In HUMAN_RECORD + _recording=False → open writer, _recording=True.
    Frames flow into a new dagger episode.
  - In HUMAN_RECORD + _recording=True → close writer, _recording=False.
    Episode is finalized; state stays HUMAN_RECORD.
  - In any other state → ignored (logged).

Multiple toggles within one (1,1) window produce multiple dagger episodes
— useful when one freedrive grip yields several distinct correction
segments. State machine cares about switches; pedal cares about which
frames are intervention=1.

Two-step button gate solves the "static prelude" problem: user opens
freedrive switches one at a time, drag only engages after BOTH are on
(meaning hands are physically on the masters and ready to drag). See
docs/deployment/strategy/dagger_implementation_plan.md §4.5.

state/action convention (KAI0 official, KAI0_ACTION_EQ_STATE=1):
  state  = puppet left[7] + puppet right[7]  (slave joint feedback)
  action = state for the 12 arm joints; the 2 gripper dims (6=L, 13=R) follow the
           master (teleop leader) grasp command when KAI0_GRIPPER_FROM_MASTER=1
           (default), falling back to slave gripper until a master topic arrives.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import threading
import time
from enum import Enum
from typing import Optional

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty, String


# ── reuse the data_manager writer (same on-disk bytes as teleop/autonomy) ──
def _bootstrap_backend_path() -> None:
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "web" / "data_manager" / "backend" / "app" / "dataset_writer.py"
        if candidate.is_file():
            sys.path.insert(0, str(candidate.parent.parent))
            return
    raise RuntimeError("could not locate web/data_manager/backend")


os.environ["KAI0_DATA_ROOT"] = os.environ.get("KAI0_RECORDING_ROOT", "/data1/DATA_IMP/KAI0")
_bootstrap_backend_path()
from app.dataset_writer import (  # noqa: E402
    CAMERAS,
    DEPTH_CAMERAS,
    EpisodeWriter,
    FPS,
    next_episode_id,
    update_info_json,
    write_episode_meta,
)


CAM_RGB_TOPIC = {
    "top_head":   "/camera_f/camera/color/image_raw",
    # mid_head = WHEELTEC UVC 第二头相机 (uvc_camera_node.py, /dev/cam_mid_head)。
    # 注意: UVC 节点发 /camera_m/color/image_raw (无 realsense 的 /camera/ 子命名)。
    "mid_head":   "/camera_m/color/image_raw",
    "hand_left":  "/camera_l/camera/color/image_raw",
    "hand_right": "/camera_r/camera/color/image_raw",
}
CAM_DEPTH_TOPIC = {
    "top_head":   "/camera_f/camera/aligned_depth_to_color/image_raw",
    "hand_left":  "/camera_l/camera/aligned_depth_to_color/image_raw",
    "hand_right": "/camera_r/camera/aligned_depth_to_color/image_raw",
}

# Slave readback (state) + master readback (action publisher when 0xFA).
SLAVE_LEFT_TOPIC  = "/puppet/joint_left"
SLAVE_RIGHT_TOPIC = "/puppet/joint_right"
MASTER_LEFT_TOPIC  = "/master/joint_left"
MASTER_RIGHT_TOPIC = "/master/joint_right"

# Control surfaces — masters in 0xFC mode subscribe to /master_controled/joint_*
# (arm_teleop_node line 135). teleop_launch remaps the per-arm names below.
MASTER_DRIVE_LEFT  = "/master_controled/joint_left"
MASTER_DRIVE_RIGHT = "/master_controled/joint_right"
MASTER_CONFIG_LEFT  = "/teach/master_config_left"
MASTER_CONFIG_RIGHT = "/teach/master_config_right"
MASTER_ENABLE_LEFT  = "/teach/master_enable_left"
MASTER_ENABLE_RIGHT = "/teach/master_enable_right"
MASTER_TEACH_LEFT   = "/teach/teach_mode_left"
MASTER_TEACH_RIGHT  = "/teach/teach_mode_right"

# Safety: master should never blindly track slave from home position. Always go
# through a known-safe intermediate pose first (matches kai0 upstream agilex
# DAgger script). Empirically chosen — both arms inside their reachable
# workspace with no risk of self-collision.
SAFE_POSE = [0.0, 0.32, -0.36, 0.0, 0.24, 0.0, 0.07]

ALIGN_TOL_RAD = 0.02       # ~1.1° per joint
ALIGN_TIMEOUT_S = 5.0
ALIGN_PUBLISH_HZ = 10.0    # 10 Hz matches upstream; 50 Hz tended to overshoot
ALIGN_DURATION_S = 3.0     # how long to publish each target before moving on
MIN_EPISODE_SEC = 1.0      # drop accidental tap-toggles (was 3.0; lowered so短 rollout 也能存)


class State(Enum):
    POLICY_RUN = "POLICY_RUN"
    ALIGNING = "ALIGNING"
    HUMAN_RECORD = "HUMAN_RECORD"
    RETURNING = "RETURNING"


def _decode_image_rgb(msg: Image) -> Optional[np.ndarray]:
    w, h, enc = msg.width, msg.height, msg.encoding
    data = bytes(msg.data)
    if enc == "rgb8":
        return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).copy()
    if enc == "bgr8":
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1])
    return None


def _decode_image_depth(msg: Image) -> Optional[np.ndarray]:
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    w, h = msg.width, msg.height
    return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w).copy()


def _to_7dim(msg: JointState) -> list[float]:
    pos = list(msg.position)[:7]
    pos += [0.0] * (7 - len(pos))
    return [float(x) for x in pos]


def _infer_task_from_ckpt(ckpt_dir: str) -> str:
    if not ckpt_dir:
        return "Task_A"
    s = ckpt_dir.lower()
    for letter in ("a", "b", "c", "d", "e"):
        if re.search(rf"\btask[_-]?{letter}\b", s) or f"/task_{letter}/" in s:
            return f"Task_{letter.upper()}"
    return "Task_A"


def _infer_prompt_from_ckpt(ckpt_dir: str) -> str:
    if not ckpt_dir:
        return "dagger correction"
    cfg_path = pathlib.Path(ckpt_dir) / "train_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
            p = cfg.get("prompt") or cfg.get("task_prompt") or cfg.get("default_prompt")
            if p:
                return str(p)
        except Exception:
            pass
    return f"dagger correction for {pathlib.Path(ckpt_dir).name}"


class DaggerRecorder(Node):
    def __init__(self) -> None:
        super().__init__("dagger_recorder")

        self.declare_parameter("task_name", "")
        self.declare_parameter("prompt", "")
        self.declare_parameter("subset", "dagger")
        self.declare_parameter("operator", "dagger")
        self.declare_parameter("checkpoint_dir", "")
        self.declare_parameter("align_tol_rad", ALIGN_TOL_RAD)
        self.declare_parameter("align_timeout_s", ALIGN_TIMEOUT_S)
        self.declare_parameter("min_episode_sec", MIN_EPISODE_SEC)
        self.declare_parameter("master_left_available", True)
        self.declare_parameter("master_right_available", True)
        # Form C dual-dataset toggle. When false, the policy-rollout (inference/)
        # side is fully disabled: no inference episode is opened/written and no
        # <task>/inference/<vN>/<date-vN>/ dir is created — only dagger/ is recorded.
        # dynamic_typing: shell→launch passes `record_inference:=false`, which
        # launch_ros serializes into the params YAML as an *unquoted* scalar →
        # rclpy loads it as BOOL, not STRING. A static STRING default would then
        # raise InvalidParameterTypeException and kill the node at startup. Allow
        # either type; the str()-coercion below normalizes bool or string alike.
        self.declare_parameter(
            "record_inference", "true",
            ParameterDescriptor(dynamic_typing=True))

        ckpt_dir = self.get_parameter("checkpoint_dir").value or ""
        task_p = self.get_parameter("task_name").value or ""
        prompt_p = self.get_parameter("prompt").value or ""
        self._task: str = task_p or _infer_task_from_ckpt(ckpt_dir)
        self._prompt: str = prompt_p or _infer_prompt_from_ckpt(ckpt_dir)
        self._subset: str = self.get_parameter("subset").value or "dagger"
        self._operator: str = self.get_parameter("operator").value or "dagger"
        self._align_tol: float = float(self.get_parameter("align_tol_rad").value)
        self._align_timeout: float = float(self.get_parameter("align_timeout_s").value)
        self._min_ep_sec: float = float(self.get_parameter("min_episode_sec").value)
        self._record_inference: bool = str(
            self.get_parameter("record_inference").value
        ).strip().lower() in ("1", "true", "yes", "on")
        self._master_available = {
            "L": str(self.get_parameter("master_left_available").value).lower()
                 in ("1", "true", "yes", "on"),
            "R": str(self.get_parameter("master_right_available").value).lower()
                 in ("1", "true", "yes", "on"),
        }
        self._teleop_sides = tuple(
            side for side in ("L", "R") if self._master_available[side])

        self._lock = threading.Lock()
        self._state: State = State.POLICY_RUN

        # Dagger writer — opened/closed by pedal toggles inside HUMAN_RECORD.
        # Multiple pedal cycles within one (1,1) window produce multiple
        # episodes. Counts intervention=1 frames per episode.
        self._writer: Optional[EpisodeWriter] = None
        self._started_at: float = 0.0
        self._wrote_frames = 0
        # Pedal flag, independent of state machine. _on_pedal_toggle is the
        # only writer to this; _on_record_tick / _on_pedal_toggle read it.
        # Frames are written to dagger writer only when state=HUMAN_RECORD
        # AND _recording=True.
        self._recording: bool = False

        # Inference writer (Form C dual-dataset) — auto-opens in POLICY_RUN
        # once slave has moved + RGB ready, finalized on takeover. Counts
        # the policy rollout frames (intervention=0).
        self._inference_writer: Optional[EpisodeWriter] = None
        self._inf_started_at: float = 0.0
        self._inf_wrote_frames = 0
        # 油门加速标识 (最小改动方案): 所有 rollout 都写单一 inference/ 数据集;
        # 每条 episode 的 meta 记 used_throttle (本段有没有踩过油门) + speed_factor
        # (本段峰值倍率, 油门段>1 / 普通段=1.0), 下游据此区分加速/普通, 不分目录。
        self._inf_subset: str = "inference"
        self._inf_speed_max: float = 1.0
        self._inf_throttle_used: bool = False

        # ── 直接采集 chunk-001 (KAI0_DIRECT_CHUNK001=1) ──
        # 一个 rollout = 一个 episode: 单 writer 从策略开跑一直开到本次尝试结束,
        # INF 段写 intervention=0 / 人接管段写 1, 空档 (ALIGNING / 未踩踏板 /
        # RETURNING) 一帧不写。dagger_frame_class 在 finalize 由 intervention 列
        # 回溯导出。产出与离线 stitch_dagger_episodes.py 完全同构, 但省掉离线
        # 重编码那一遍, 且不再落 chunk-000 / inference 两份原始数据。
        # 关掉 (=0) 则完全走旧 Form C 双数据集路径, 逐比特不变。
        self._direct_c1 = os.environ.get("KAI0_DIRECT_CHUNK001", "0") == "1"
        self._roll_writer: Optional[EpisodeWriter] = None
        self._roll_started_at: float = 0.0
        self._roll_frames = 0        # 送进 writer 的 tick 数 (含被段内 trim 掉的)
        self._roll_iv_frames = 0     # 其中 intervention=1 的
        self._roll_takeovers = 0     # 本 rollout 发生过几次人接管
        self._roll_speed_max = 1.0
        self._roll_throttle_used = False

        # ── Per-rollout boundary + inference↔dagger alignment ──
        # One "rollout" = one autonomous task attempt (cloth fold, pick-place,
        # wipe, …). The /dagger/rollout_next button toggles a pause between
        # rollouts (finalize inference as completed → execute=false → operator
        # resets the scene → press again → new inference ep + execute=true, which
        # flushes RTC on the policy side; the model is NOT reloaded).
        # Alignment keys stamped into BOTH inference + dagger episode meta:
        #   rollout_id    — shared by every episode (inference + dagger) of one fold.
        #   takeover_id   — increments per takeover; an inference segment cut by a
        #                   takeover records ends_takeover_id, and the dagger
        #                   correction recorded during that takeover records the
        #                   same takeover_id → the two are paired for RECAP/IWR.
        self._rollout_paused = False
        self._rollout_id = 0
        self._takeover_id = 0
        self._cur_takeover_id: Optional[int] = None

        self._rgb: dict[str, Optional[np.ndarray]] = {c: None for c in CAMERAS}
        self._depth: dict[str, Optional[np.ndarray]] = {c: None for c in DEPTH_CAMERAS}
        self._q_slave_left:  list[float] = [0.0] * 7
        self._q_slave_right: list[float] = [0.0] * 7
        self._q_master_left:  list[float] = [0.0] * 7
        self._q_master_right: list[float] = [0.0] * 7
        # Whether a master (teleop leader) JointState has actually arrived — gates
        # the V3 gripper-from-master action override (fall back to slave gripper
        # until the master topic is live). See KAI0_GRIPPER_FROM_MASTER below.
        self._got_master_left = False
        self._got_master_right = False
        self._grip_from_master = os.environ.get("KAI0_GRIPPER_FROM_MASTER", "1") == "1"
        self._align_target_left:  Optional[list[float]] = None
        self._align_target_right: Optional[list[float]] = None

        # Physical-button raw state (5 Hz polled by arm_master_servo_node).
        self._button_left = False
        self._button_right = False
        # Previous level — used for edge detection. Without this, level-
        # triggered _on_button fires repeatedly while a switch is held ON,
        # spamming takeover attempts (one per 200 ms poll). See Phase D1
        # post-mortem in dagger_implementation_plan.md.
        self._prev_any_pressed = False
        self._prev_all_available_pressed = False

        # Grace period: don't accept takeover until slave has moved from its
        # boot zero pose. Some freedrive switches default to ON at power-up,
        # which would otherwise spawn a takeover before policy starts driving
        # the slave (slave at zero → _do_takeover aborts → loop). Cleared
        # once any slave joint exceeds 0.01 rad from zero.
        self._slave_seen_nonzero = False

        # Startup gate: if a freedrive switch is already ON when dagger_recorder
        # boots, the FIRST button message arrives with msg.data=True. Without
        # this gate, that gets treated as a rising edge → premature takeover
        # before policy has even finished loading (policy was loading JAX,
        # received execute=False mid-load, then went to OBSERVE on init —
        # observed in /tmp/dagger_step2_log.txt). We require seeing at least
        # one "all-OFF" message first, proving the switches are actually being
        # toggled by the operator after startup, not just held over from a
        # previous run / left high at power-up.
        self._seen_off_after_boot = False

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for cam, topic in CAM_RGB_TOPIC.items():
            self.create_subscription(
                Image, topic,
                lambda msg, k=cam: self._on_rgb(k, msg),
                sensor_qos,
            )
        for cam in DEPTH_CAMERAS:
            topic = CAM_DEPTH_TOPIC.get(cam)
            if topic:
                self.create_subscription(
                    Image, topic,
                    lambda msg, k=cam: self._on_depth(k, msg),
                    sensor_qos,
                )

        self.create_subscription(JointState, SLAVE_LEFT_TOPIC,
                                 lambda m: self._on_slave("L", m), 10)
        self.create_subscription(JointState, SLAVE_RIGHT_TOPIC,
                                 lambda m: self._on_slave("R", m), 10)
        self.create_subscription(JointState, MASTER_LEFT_TOPIC,
                                 lambda m: self._on_master("L", m), 10)
        self.create_subscription(JointState, MASTER_RIGHT_TOPIC,
                                 lambda m: self._on_master("R", m), 10)

        self.create_subscription(Bool, "/dagger/takeover", self._on_takeover, 1)
        # Single-button per-fold boundary (toggle: end-fold ↔ start-next-fold).
        self.create_subscription(Empty, "/dagger/rollout_next", self._on_rollout_next, 1)
        # Per-arm physical-button state (published by arm_master_servo_node).
        # dagger_launch remaps these to /master_button_left and /master_button_right.
        self.create_subscription(Bool, "/master_button_left",
                                  lambda m: self._on_button("L", m), 5)
        self.create_subscription(Bool, "/master_button_right",
                                  lambda m: self._on_button("R", m), 5)
        # USB pedal toggle (published by dagger_pedal_node on F3 release).
        # Each Empty event flips the _recording flag inside HUMAN_RECORD.
        self.create_subscription(Empty, "/dagger/pedal_toggled",
                                  self._on_pedal_toggle, 5)
        # Explicit start/save/discard commands (web/dagger_manager 三按钮),
        # mirroring start_data_collect.sh's recorder. String in {start,save,
        # discard}. Pedal stays a start↔save toggle; discard is web-only.
        self.create_subscription(String, "/dagger/record_cmd",
                                  self._on_record_cmd, 5)
        # V2 油门: policy_inference_node (session 进程) latched 广播的全局速度倍率.
        # 落进 inference/ episode meta, 让高速 rollout 数据可复现. transient_local 保证
        # 本 infra 进程晚 join 也能收到最后一次值. 缺省 1.0 (未开油门 = 原速采集).
        self._cur_speed_factor = 1.0
        from std_msgs.msg import Float32 as _Float32
        # 用模块级 QoSProfile/QoSDurabilityPolicy (line 72); 不要在此局部 re-import
        # QoSProfile — 会把它变成 __init__ 局部变量, 令上面 sensor_qos=QoSProfile(...)
        # 触发 UnboundLocalError。
        _latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(_Float32, "/policy/speed_factor",
                                 self._on_speed_factor, _latched)

        self.pub_execute = self.create_publisher(Bool, "/policy/execute", 1)
        # State machine snapshot for web/dagger_manager (latched, so a late
        # subscriber gets the current state immediately). One String message
        # per transition; consumers compare against State enum names.
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_state = self.create_publisher(String, "/dagger/state", latched)
        self.pub_state.publish(String(data=self._state.value))
        # Recording flag: separate from state so consumers (web UI, logging)
        # can render "in dagger mode but not actively writing" distinctly
        # from "actively writing a dagger episode".
        self.pub_recording = self.create_publisher(Bool, "/dagger/recording", latched)
        self.pub_recording.publish(Bool(data=self._recording))
        self.pub_master_available_left = self.create_publisher(
            Bool, "/dagger/master_available_left", latched)
        self.pub_master_available_right = self.create_publisher(
            Bool, "/dagger/master_available_right", latched)
        self.pub_master_available_left.publish(
            Bool(data=self._master_available["L"]))
        self.pub_master_available_right.publish(
            Bool(data=self._master_available["R"]))
        # Rollout pause flag (latched): True = between rollouts, waiting for the
        # operator to reset the scene and press "start next". Lets the web UI show
        # whether the next button press will END the current rollout or START a new one.
        self.pub_rollout_paused = self.create_publisher(Bool, "/dagger/rollout_paused", latched)
        self.pub_rollout_paused.publish(Bool(data=self._rollout_paused))
        self.pub_drive_left  = self.create_publisher(JointState, MASTER_DRIVE_LEFT, 10)
        self.pub_drive_right = self.create_publisher(JointState, MASTER_DRIVE_RIGHT, 10)
        self.pub_cfg_left  = self.create_publisher(String, MASTER_CONFIG_LEFT, 1)
        self.pub_cfg_right = self.create_publisher(String, MASTER_CONFIG_RIGHT, 1)
        # Match upstream agilex DAgger script: explicit re-enable + teach_mode pubs
        from std_msgs.msg import Int32  # local import — Int32 only used here
        self.pub_enable_left  = self.create_publisher(Bool, MASTER_ENABLE_LEFT, 10)
        self.pub_enable_right = self.create_publisher(Bool, MASTER_ENABLE_RIGHT, 10)
        self.pub_teach_left   = self.create_publisher(Int32, MASTER_TEACH_LEFT, 10)
        self.pub_teach_right  = self.create_publisher(Int32, MASTER_TEACH_RIGHT, 10)
        self._Int32 = Int32

        self.create_timer(1.0 / FPS, self._on_record_tick)

        # No mirror loop — master_servo subscribes /master/joint_* directly,
        # so master physically tracks whatever drives the slave (policy or
        # master's own encoder publish) automatically.

        self.get_logger().info(
            f"dagger_recorder ready: task={self._task} subset={self._subset} "
            f"prompt={self._prompt!r} fps={FPS} "
            f"record_inference={'ON' if self._record_inference else 'OFF (dagger-only)'}\n"
            f"  masters={list(self._teleop_sides) or ['none (rollout-only)']} "
            f"state={self._state.value}"
        )

    def _publish_state(self) -> None:
        """Latch-publish current state to /dagger/state for web UI consumers.
        Safe to call inside or outside self._lock — only reads self._state."""
        try:
            self.pub_state.publish(String(data=self._state.value))
        except Exception:
            pass

    def _publish_recording(self) -> None:
        """Latch-publish current recording flag to /dagger/recording."""
        try:
            self.pub_recording.publish(Bool(data=self._recording))
        except Exception:
            pass

    def _control_capability_meta(self) -> dict:
        """Episode-level contract for reconstructing per-side supervision.

        For every frame with intervention=1, available sides are human-driven;
        unavailable sides are HOLD and must be masked out of expert loss.
        """
        left = self._master_available["L"]
        right = self._master_available["R"]
        return {
            "available_masters": [s.lower() for s in self._teleop_sides],
            "master_available_left": left,
            "master_available_right": right,
            "teleop_scope": "dual" if left and right else (
                "left" if left else ("right" if right else "none")),
            "uncontrolled_side_during_teleop": "hold",
            "human_supervision_mask": [1] * 7 + [0] * 7 if left and not right else (
                [0] * 7 + [1] * 7 if right and not left else (
                    [1] * 14 if left and right else [0] * 14)),
        }

    # ── sensor callbacks ──
    def _on_rgb(self, cam: str, msg: Image) -> None:
        arr = _decode_image_rgb(msg)
        if arr is None:
            return
        with self._lock:
            self._rgb[cam] = arr

    def _on_depth(self, cam: str, msg: Image) -> None:
        arr = _decode_image_depth(msg)
        if arr is None:
            return
        with self._lock:
            self._depth[cam] = arr

    def _on_slave(self, side: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        with self._lock:
            if side == "L":
                self._q_slave_left = pos
            else:
                self._q_slave_right = pos
            # Grace period: once any slave joint has clearly moved off zero,
            # button presses can trigger takeover. Until then, freedrive
            # switches held ON since boot are ignored.
            if not self._slave_seen_nonzero:
                if (any(abs(x) > 0.01 for x in self._q_slave_left[:6]) or
                        any(abs(x) > 0.01 for x in self._q_slave_right[:6])):
                    self._slave_seen_nonzero = True

    def _on_master(self, side: str, msg: JointState) -> None:
        pos = _to_7dim(msg)
        with self._lock:
            if side == "L":
                self._q_master_left = pos
                self._got_master_left = True
            else:
                self._q_master_right = pos
                self._got_master_right = True

    def _on_button(self, side: str, msg: Bool) -> None:
        """Per-arm freedrive-button state from arm_master_servo's teach_status poll.

        Edge-triggered (rising/falling) so that level-held switches don't spam
        callbacks. The 5 Hz polling in arm_master_servo means a held-ON switch
        publishes True every 200 ms — without edge detection that would spawn
        a takeover thread every tick.

        Capability-aware rule: all *available* masters must be in teach mode to
        enter teleop, and all available masters must leave teach mode to hand
        control back.  With one available master this naturally becomes a
        single-switch workflow; with no masters button events are ignored.

        Grace period: takeover ignored until slave has moved off boot zero,
        protecting against boot-time switch-already-ON race.
        """
        pressed = bool(msg.data)
        with self._lock:
            if side == "L":
                self._button_left = pressed
            else:
                self._button_right = pressed
            levels = {"L": self._button_left, "R": self._button_right}
            available_pressed = [levels[s] for s in self._teleop_sides]
            any_pressed = any(available_pressed)
            all_pressed = bool(available_pressed) and all(available_pressed)
            all_off = bool(available_pressed) and not any_pressed
            prev_any = self._prev_any_pressed
            prev_all = self._prev_all_available_pressed
            all_rising = all_pressed and not prev_all
            all_falling = all_off and prev_any
            self._prev_any_pressed = any_pressed
            self._prev_all_available_pressed = all_pressed
            cur = self._state
            grace = self._slave_seen_nonzero
            # First all-OFF message arms the rising-edge detector.
            if not any_pressed and not self._seen_off_after_boot:
                self._seen_off_after_boot = True
            seen_off = self._seen_off_after_boot

        if not self._teleop_sides:
            return
        if all_rising and cur in (State.POLICY_RUN, State.ALIGNING) and self._rollout_paused:
            self.get_logger().info(f"[button] {side} rising → IGNORED (rollout paused for cloth reset)")
            return
        if all_rising and cur in (State.POLICY_RUN, State.ALIGNING):
            if not seen_off:
                self.get_logger().warn(
                    f"[button] {side} rising → IGNORED (freedrive switches were "
                    "ALREADY ON at dagger_recorder boot; release both switches "
                    "first, then re-engage to trigger takeover)"
                )
                return
            if not grace:
                self.get_logger().warn(
                    f"[button] {side} rising → IGNORED (slave still at boot "
                    "zero pose; freedrive switches must come ON after policy "
                    "starts driving slave)"
                )
                return
            self.get_logger().info(
                f"[button] {side} rising → takeover "
                f"(L={self._button_left} R={self._button_right})"
            )
            threading.Thread(target=self._do_takeover, daemon=True).start()
        elif any_pressed and cur == State.POLICY_RUN and not all_pressed:
            # A master servo starts publishing as soon as its own switch enters
            # teach mode. Halt policy immediately to avoid two publishers while
            # waiting for the other available master.
            self.get_logger().warn(
                f"[button] waiting for all available masters "
                f"(L={self._button_left} R={self._button_right}); policy halted")
            self.pub_execute.publish(Bool(data=False))
            with self._lock:
                self._state = State.ALIGNING
            self._publish_state()
        elif all_falling and cur == State.ALIGNING:
            self.get_logger().info("[button] alignment cancelled; resume policy")
            self.pub_execute.publish(Bool(data=True))
            with self._lock:
                self._state = State.POLICY_RUN
            self._publish_state()
        elif all_falling and cur == State.HUMAN_RECORD:
            self.get_logger().info(
                f"[button] {side} falling → handback "
                f"(L={self._button_left} R={self._button_right})"
            )
            threading.Thread(target=self._do_handback, daemon=True).start()

    # ── /dagger/takeover edge handler ──
    def _on_speed_factor(self, msg) -> None:
        """V2 油门: 缓存 policy 广播的当前速度倍率, 供 inference episode meta 记录."""
        try:
            sf = float(msg.data)
        except (TypeError, ValueError):
            return
        if abs(sf - self._cur_speed_factor) > 1e-3:
            self.get_logger().info(f"[SPEED] inference 采集速度倍率 → {sf:.2f}")
        self._cur_speed_factor = sf

    def _on_takeover(self, msg: Bool) -> None:
        want = bool(msg.data)
        if want and not self._teleop_sides:
            self.get_logger().warn(
                "/dagger/takeover=true ignored: no master arms are available")
            return
        with self._lock:
            cur = self._state
            levels = {"L": self._button_left, "R": self._button_right}
            all_on = bool(self._teleop_sides) and all(
                levels[s] for s in self._teleop_sides)
            all_off = bool(self._teleop_sides) and not any(
                levels[s] for s in self._teleop_sides)
        if want and not all_on:
            self.get_logger().warn(
                "/dagger/takeover=true ignored: all available masters must be in teach mode")
            return
        if not want and cur == State.HUMAN_RECORD and not all_off:
            self.get_logger().warn(
                "/dagger/takeover=false ignored: all available masters must leave teach mode")
            return
        if want and cur == State.POLICY_RUN:
            threading.Thread(target=self._do_takeover, daemon=True).start()
        elif (not want) and cur == State.HUMAN_RECORD:
            threading.Thread(target=self._do_handback, daemon=True).start()
        else:
            self.get_logger().warn(
                f"/dagger/takeover={want} ignored in state={cur.value}"
            )

    # ── /dagger/pedal_toggled handler ──
    def _on_pedal_toggle(self, _msg: Empty) -> None:
        """Pedal press (F3 release on default HID 0483:5750).

        Toggles the dagger writer open/closed WITHOUT touching the state
        machine. Each press in HUMAN_RECORD with _recording=False starts a
        new episode; press with _recording=True finalizes it. State stays
        HUMAN_RECORD across pedal cycles — only switches drive transitions.

        Ignored in POLICY_RUN / ALIGNING / RETURNING.
        """
        with self._lock:
            cur = self._state
            recording = self._recording
        if cur != State.HUMAN_RECORD:
            self.get_logger().info(
                f"[pedal] ignored in state={cur.value} "
                "(pedal only meaningful in HUMAN_RECORD)"
            )
            return
        # Pedal = start↔save toggle (no discard; discard is web-only).
        if not recording:
            self._start_recording(src="pedal")
        else:
            self._save_recording(src="pedal")

    # ── /dagger/record_cmd handler (web 三按钮: start / save / discard) ──
    def _on_record_cmd(self, msg: String) -> None:
        """Explicit recording command, mirroring start_data_collect.sh's
        recorder.start / save / discard. State machine is untouched — these
        only drive the dagger episode writer inside HUMAN_RECORD."""
        cmd = (msg.data or "").strip().lower()
        if cmd == "start":
            self._start_recording(src="cmd")
        elif cmd == "save":
            self._save_recording(src="cmd")
        elif cmd == "discard":
            self._discard_recording(src="cmd")
        else:
            self.get_logger().warn(
                f"[record] unknown cmd '{cmd}' (want start|save|discard)"
            )

    # ── recording helpers (shared by pedal + record_cmd) ──
    def _start_recording(self, src: str = "cmd") -> bool:
        """Open a dagger episode (only in HUMAN_RECORD, only if not already
        recording). Returns True on success."""
        with self._lock:
            cur = self._state
            recording = self._recording
        if cur != State.HUMAN_RECORD:
            self.get_logger().info(
                f"[{src}] start ignored in state={cur.value} (need HUMAN_RECORD)")
            return False
        if recording:
            self.get_logger().info(f"[{src}] start ignored — already recording")
            return False
        if self._direct_c1:
            # 段边界而非新 episode: 切人接管 profile, 起手静止段一帧不留
            # (离线 class 3 全裁)。writer 必须已经开着 —— rollout 起于 POLICY_RUN。
            with self._lock:
                ok = self._roll_writer is not None
            if not ok:
                self.get_logger().warn(
                    f"[{src}] start ignored — 本 rollout 的 writer 还没开 "
                    "(策略一帧都没录到?)")
                return False
            self.get_logger().info(f"[{src}] start human segment (起手静止全裁)")
            self._roll_segment("human_seg_begin")
            with self._lock:
                self._recording = True
            self._publish_recording()
            return True

        self.get_logger().info(f"[{src}] open dagger episode (recording=True)")
        self._open_episode()
        with self._lock:
            ok = self._writer is not None
            if ok:
                self._recording = True
            else:
                self.get_logger().warn(
                    f"[{src}] writer init failed; recording stays False")
        self._publish_recording()
        return ok

    def _save_recording(self, src: str = "cmd") -> bool:
        """Finalize + keep the current dagger episode."""
        with self._lock:
            recording = self._recording
        if not recording:
            self.get_logger().info(f"[{src}] save ignored — not recording")
            return False
        if self._direct_c1:
            self.get_logger().info(f"[{src}] close human segment (尾部静止全裁, recording=False)")
            self._roll_segment("human_seg_end")
        else:
            self.get_logger().info(f"[{src}] save dagger episode (finalize, recording=False)")
            self._close_episode()
        with self._lock:
            self._recording = False
        self._publish_recording()
        return True

    def _discard_recording(self, src: str = "cmd") -> bool:
        """Abort the current dagger episode — delete partial files, keep none."""
        with self._lock:
            recording = self._recording
        if not recording:
            self.get_logger().info(f"[{src}] discard ignored — not recording")
            return False
        if self._direct_c1:
            self._discard_rollout_episode(why=f"{src} discard")
        else:
            self.get_logger().info(f"[{src}] discard dagger episode (abort, recording=False)")
            self._discard_episode()
        with self._lock:
            self._recording = False
        self._publish_recording()
        return True

    # ── mirror loop (policy phase): continuously publish slave's encoder pose
    # to /master_controled/joint_* so the master's arm_master_servo_node drives
    # the master arm via JointCtrl to mirror slave. Visual feedback: when
    # policy moves slave, master moves in sync.
    # mirror_loop removed: master_servo subscribes /master/joint_* directly,
    # so the action stream published by policy_inference drives both slave
    # (via arm_reader mode=1) and master (via arm_master_servo subscribe state).

    # ── state transitions ──
    def _do_takeover(self) -> None:
        """POLICY_RUN → HUMAN_RECORD.

        Architecture: master arms are flashed as 0xFC followers (CAN-controllable).
        arm_master_servo_node accepts /master/enable Bool: True=control state
        (motors hold, accept JointCtrl), False=drag state (motors free, publish
        encoder to /master/joint_*).

        During policy: mirror loop publishes slave_pose → master mirrors slave.
        Takeover sequence:
          1. halt policy (/policy/execute=false)
          2. validate slave pose is non-zero
          3. master_enable=False → arm_master_servo DisableArm + start publishing
             /master/joint_* from encoder. slave's arm_reader follows.
          4. open episode
        """
        with self._lock:
            self._state = State.ALIGNING
        self._publish_state()

        # 1) halt policy publishing /master/joint_* + finalize inference ep.
        # terminal="intervention": this rollout FAILED (operator took over) →
        # success=False + intervention_frame_index recorded for RECAP/IWR.
        self.get_logger().info("[TAKEOVER] 1/4 halt policy + close inference episode")
        self.pub_execute.publish(Bool(data=False))
        time.sleep(0.5)
        # New takeover id pairs the inference segment we're about to close
        # (ends_takeover_id) with the dagger correction recorded next (takeover_id).
        with self._lock:
            self._takeover_id += 1
            self._cur_takeover_id = self._takeover_id
            if self._direct_c1:
                self._roll_takeovers += 1
        if self._direct_c1:
            # 连续单集: 不关 writer, 只冲掉待定缓冲且【不做尾裁】。这段静止尾巴是
            # "模型卡死/回折" 的失败先兆 (preintv 负样本), 恰恰是最该留的帧。
            # (滑行伪影不存在 — 上面已切 ALIGNING, _on_record_tick 从此静默。)
            self._roll_segment("flush_keep")
        else:
            self._close_inference_episode(terminal="intervention")

        # 2) validate slave pose
        with self._lock:
            tl = list(self._q_slave_left)
            tr = list(self._q_slave_right)
        zero_l = all(abs(x) < 1e-4 for x in tl[:6])
        zero_r = all(abs(x) < 1e-4 for x in tr[:6])
        if zero_l or zero_r:
            self.get_logger().error(
                f"[TAKEOVER] ABORT — slave pose empty (L_zero={zero_l} R_zero={zero_r})"
            )
            self.pub_execute.publish(Bool(data=True))
            with self._lock:
                self._state = State.POLICY_RUN
            self._publish_state()
            return
        self.get_logger().info(
            f"[TAKEOVER] 2/4 slave OK  L={[round(x,3) for x in tl[:6]]}  R={[round(x,3) for x in tr[:6]]}"
        )

        # Master should already be at slave's pose from the mirror loop. Now
        # we transition master from "control" state (motors hold + accept
        # JointCtrl) to "drag" state (motors free + encoder publishes).

        # 3) master_enable=False → DisableArm + start encoder publishing
        self.get_logger().info(
            f"[TAKEOVER] 3/4 switch available masters {self._teleop_sides} to drag state")
        for _ in range(3):
            if self._master_available["L"]:
                self.pub_enable_left.publish(Bool(data=False))
            if self._master_available["R"]:
                self.pub_enable_right.publish(Bool(data=False))
            time.sleep(0.1)
        time.sleep(1.2)  # let DisableArm settle and encoder publisher start

        # 4) drag mode ready; pedal will toggle the writer independently
        self.get_logger().info(
            "[TAKEOVER] 4/4 drag mode active — pedal controls recording"
        )
        with self._lock:
            self._state = State.HUMAN_RECORD
            self._recording = False
        self._publish_state()
        self._publish_recording()
        self.get_logger().info(
            "[TAKEOVER] DONE — master is free to drag. "
            "Press pedal (F3) to start/stop recording; toggle switches OFF to handback."
        )

    def _do_handback(self) -> None:
        """HUMAN_RECORD → POLICY_RUN.

        1. finalize episode
        2. master_enable=True → arm_master_servo EnableArm + CAN_CTRL + stop encoder publish
        3. /policy/execute=true → policy resumes
        4. Mirror loop resumes (state=POLICY_RUN), master follows slave again
        """
        with self._lock:
            self._state = State.RETURNING
            was_recording = self._recording
            had_writer = self._writer is not None
            self._recording = False
        self._publish_state()
        self._publish_recording()

        if self._direct_c1:
            # 人接管段在此结束 → 尾部静止段全裁 (离线 class 4)。writer 不关。
            if was_recording:
                self.get_logger().info("[HANDBACK] 1/3 close human segment (尾部静止全裁)")
                self._roll_segment("human_seg_end")
            else:
                self.get_logger().info(
                    "[HANDBACK] 1/3 skip segment close (pedal never started a segment)")
        # Close writer only if pedal had opened one (idempotent if not).
        elif had_writer:
            self.get_logger().info("[HANDBACK] 1/3 finalize dagger episode")
            self._close_episode()
        else:
            self.get_logger().info(
                "[HANDBACK] 1/3 skip episode close (pedal never opened a writer)"
            )

        self.get_logger().info("[HANDBACK] 2/3 master_enable=True (EnableArm + CAN_CTRL)")
        for _ in range(3):
            if self._master_available["L"]:
                self.pub_enable_left.publish(Bool(data=True))
            if self._master_available["R"]:
                self.pub_enable_right.publish(Bool(data=True))
            time.sleep(0.1)
        time.sleep(2.0)  # EnablePiper loop in arm_master_servo can take ~1s

        self.get_logger().info("[HANDBACK] 3/3 resume policy")
        self.pub_execute.publish(Bool(data=True))

        with self._lock:
            self._state = State.POLICY_RUN
        self._publish_state()
        # Open a fresh inference episode for the next policy run (Form C).
        # _on_record_tick will lazy-confirm slave+rgb readiness before the
        # first write, so opening here is safe even if cameras momentarily
        # drop or slave is still settling after handback.
        # 直接采集模式下 writer 从未关闭 → 什么都不用做, 下一 tick 继续往同一集写
        # intervention=0。也【不】重新武装 front-trim: 策略是在任务中途接着跑的,
        # 起手低速属于真实行为, 不是遥操伪影 (离线 classify_segment 对 inf 段同样不裁)。
        if not self._direct_c1:
            self._open_inference_episode()
        self.get_logger().info(
            "[HANDBACK] DONE — policy running, master mirroring slave. "
            "Toggle on to record next."
        )

    def _publish_drive(self, ql: list[float], qr: list[float]) -> None:
        now = self.get_clock().now().to_msg()
        msg_l = JointState()
        msg_l.header.stamp = now
        msg_l.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg_l.position = ql
        self.pub_drive_left.publish(msg_l)
        msg_r = JointState()
        msg_r.header.stamp = now
        msg_r.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        msg_r.position = qr
        self.pub_drive_right.publish(msg_r)

    # ── episode lifecycle (dagger = human-correction segments) ──
    def _open_episode(self) -> None:
        """Open the dagger episode writer (Form C: human side)."""
        try:
            ep = next_episode_id(self._task, self._subset)
            writer = EpisodeWriter(
                task=self._task, subset=self._subset, ep=ep,
                prompt=self._prompt, template_id="dagger", operator=self._operator,
            )
        except Exception as e:
            self.get_logger().error(f"dagger writer init failed: {e}")
            return
        with self._lock:
            self._writer = writer
            self._started_at = time.time()
            self._wrote_frames = 0
        self.get_logger().info(
            f"  dagger ep={ep} → {writer.root}"
        )

    def _close_episode(self) -> None:
        with self._lock:
            writer = self._writer
            started_at = self._started_at
            wrote = self._wrote_frames
            self._writer = None
        if writer is None:
            return
        duration = time.time() - started_at

        if duration < self._min_ep_sec or wrote < int(self._min_ep_sec * FPS):
            self.get_logger().warn(
                f"  dagger episode too short ({duration:.1f}s, {wrote} frames) — DROPPING"
            )
            try:
                writer.abort()
            except Exception as e:
                self.get_logger().error(f"abort failed: {e}")
            return

        try:
            writer.finalize()
            # Alignment: this correction pairs with the inference segment cut by
            # the same takeover (matching takeover_id / ends_takeover_id) and
            # belongs to the same fold (rollout_id).
            write_episode_meta(
                writer, duration, success=True,
                note=f"dagger correction ({wrote} frames @ {FPS} Hz)",
                scene_tags=[],
                extra={"intervention": 1, "rollout_id": self._rollout_id,
                       "takeover_id": self._cur_takeover_id,
                       **self._control_capability_meta()},
            )
            update_info_json(self._task, self._subset)
            self.get_logger().info(
                f"  saved dagger ep={writer.ep} frames={wrote} duration={duration:.1f}s "
                f"→ {writer.root}"
            )
        except Exception as e:
            self.get_logger().error(f"finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass

    def _discard_episode(self) -> None:
        """Abort the dagger writer WITHOUT finalizing — deletes the half-written
        parquet/mp4 (mirrors data_manager recorder.discard)."""
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            return
        try:
            writer.abort()
            self.get_logger().info(
                f"  discarded dagger ep={writer.ep} (partial files deleted)")
        except Exception as e:
            self.get_logger().error(f"discard abort failed: {e}")

    # ── inference episode lifecycle (Form C: policy rollout side) ──
    def _open_inference_episode(self, subset: Optional[str] = None) -> None:
        """Open an inference (policy-rollout) episode writer under <task>/inference/.

        单一 inference/ 数据集: 所有 rollout (加速/普通) 都写这里, 是否加速由 episode
        meta 的 used_throttle 标识 (见 _on_record_tick / _close_inference_episode)。
        subset 参数保留兼容, 默认 "inference"。
        """
        if not self._record_inference:
            return  # inference recording disabled by config
        subset = subset or "inference"
        try:
            ep = next_episode_id(self._task, subset)
            writer = EpisodeWriter(
                task=self._task, subset=subset, ep=ep,
                prompt=self._prompt, template_id="inference",
                operator=self._operator,
            )
        except Exception as e:
            self.get_logger().error(f"inference writer init failed: {e}")
            return
        with self._lock:
            self._inference_writer = writer
            self._inf_subset = subset
            self._inf_started_at = time.time()
            self._inf_wrote_frames = 0
            self._inf_speed_max = self._cur_speed_factor
            self._inf_throttle_used = self._cur_speed_factor > 1.0 + 1e-3
        self.get_logger().info(
            f"  inference ep={ep} → {writer.root}"
        )

    def _close_inference_episode(self, terminal: str = "session_end") -> None:
        """Finalize the current inference (policy-rollout) episode.

        terminal cause drives the success label — CRITICAL for the RECAP /
        advantage pipeline that consumes inference/ (the whole reason Form C
        records it). A rollout cut by a human takeover is a FAILED rollout (the
        operator intervened *because* the policy was failing): never claim a
        success we cannot verify.
          - "intervention" → success=False; the tail is the failure region that
            led to rescue. Record intervention_frame_index so the advantage
            estimator / IWR can locate / down-weight it.
          - "session_end"  → success=False (unverified — could be mid-task).
          - "completed"    → success=True; ONLY for an explicit success signal
            (future auto-detector or operator confirmation), never the default.
        """
        with self._lock:
            writer = self._inference_writer
            started_at = self._inf_started_at
            wrote = self._inf_wrote_frames
            self._inference_writer = None
        if writer is None:
            return
        duration = time.time() - started_at

        if duration < self._min_ep_sec or wrote < int(self._min_ep_sec * FPS):
            self.get_logger().warn(
                f"  inference episode too short ({duration:.1f}s, {wrote} frames) — DROPPING"
            )
            try:
                writer.abort()
            except Exception as e:
                self.get_logger().error(f"abort failed: {e}")
            return

        if terminal == "completed":
            success = True
            note = f"policy rollout COMPLETED ({wrote} frames @ {FPS} Hz)"
            extra = {"terminal": "completed", "rollout_id": self._rollout_id}
        elif terminal == "intervention":
            success = False
            note = (f"policy rollout TERMINATED BY INTERVENTION @ frame {wrote} "
                    f"(failed; {wrote} frames @ {FPS} Hz)")
            # ends_takeover_id pairs this failed segment with the dagger correction
            # recorded next (same takeover_id) — for RECAP/IWR credit assignment.
            extra = {"terminal": "intervention", "intervention_frame_index": wrote,
                     "rollout_id": self._rollout_id, "ends_takeover_id": self._cur_takeover_id}
        else:  # session_end / unknown — not a verified success
            success = False
            note = f"policy rollout (session_end, success unverified; {wrote} frames @ {FPS} Hz)"
            extra = {"terminal": "session_end", "rollout_id": self._rollout_id}

        # 油门加速标识: used_throttle=本段有没有踩过油门 (整段标记); speed_factor=本段峰值
        # 倍率 (踩过 >1, 没踩 =1.0)。下游据此区分加速/普通数据, 无需分目录。
        extra = {**extra, **self._control_capability_meta(),
                 "used_throttle": bool(self._inf_throttle_used),
                 "speed_factor": round(float(self._inf_speed_max), 3)}
        try:
            writer.finalize()
            write_episode_meta(
                writer, duration, success=success,
                note=note, scene_tags=[], extra=extra,
            )
            update_info_json(self._task, self._inf_subset)
            self.get_logger().info(
                f"  saved inference ep={writer.ep} frames={wrote} duration={duration:.1f}s "
                f"→ {writer.root}"
            )
        except Exception as e:
            self.get_logger().error(f"inference finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass

    # ── rollout episode lifecycle (直接采集 chunk-001) ──
    def _open_rollout_episode(self) -> None:
        """Open THE writer for this rollout. Stays open across takeover/handback —
        the state machine only decides which intervention flag each tick carries and
        which ticks are dropped entirely."""
        try:
            ep = next_episode_id(self._task, self._subset, chunk=1)
            writer = EpisodeWriter(
                task=self._task, subset=self._subset, ep=ep,
                prompt=self._prompt, template_id="stitched_dagger",
                operator=self._operator,
                chunk=1, frame_class=True, record_depth=False,
            )
        except Exception as e:
            self.get_logger().error(f"rollout writer init failed: {e}")
            return
        with self._lock:
            self._roll_writer = writer
            self._roll_started_at = time.time()
            self._roll_frames = 0
            self._roll_iv_frames = 0
            self._roll_takeovers = 0
            self._roll_speed_max = self._cur_speed_factor
            self._roll_throttle_used = self._cur_speed_factor > 1.0 + 1e-3
        self.get_logger().info(
            f"  rollout ep={ep} (chunk-001) → {writer.root}")

    def _roll_segment(self, op: str) -> None:
        """Forward a segment-boundary op to the open rollout writer (no-op if none)."""
        with self._lock:
            writer = self._roll_writer
        if writer is None:
            return
        try:
            writer.segment_control(op)
        except Exception as e:
            self.get_logger().error(f"segment_control({op}) failed: {e}")

    def _close_rollout_episode(self, terminal: str = "session_end") -> None:
        """Finalize the rollout episode. terminal drives success exactly like the
        old inference path: only an explicit rollout_next press means completed."""
        with self._lock:
            writer = self._roll_writer
            started_at = self._roll_started_at
            wrote = self._roll_frames
            iv = self._roll_iv_frames
            n_takeover = self._roll_takeovers
            throttle = self._roll_throttle_used
            speed_max = self._roll_speed_max
            self._roll_writer = None
        if writer is None:
            return
        duration = time.time() - started_at

        min_frames = int(self._min_ep_sec * FPS)
        if duration < self._min_ep_sec or wrote < min_frames:
            self.get_logger().warn(
                f"  rollout episode too short ({duration:.1f}s, {wrote} captured frames) — DROPPING")
            try:
                writer.abort()
            except Exception as e:
                self.get_logger().error(f"abort failed: {e}")
            return

        try:
            writer.finalize()
            kept = writer.frame_count
            # 裁剪后长度闸门: 上面用的是【捕获帧数】wrote, 但段边界的物理裁剪
            # (human_seg_begin/end 起手迟疑+静止尾, front/tail-trim) 可能把整段
            # 削到 min 以下 —— kept 只有 finalize() 跑完才知道。捕获够、裁完不够的
            # 退化段 (如全是遥操迟疑/卡死静止被裁光) 到此丢弃, 否则会像 ep044 那样
            # 落一条 15 帧的垃圾 episode 进 meta。abort() 删掉刚写的 parquet+mp4。
            if kept < min_frames:
                self.get_logger().warn(
                    f"  rollout episode too short AFTER trim ({kept} kept of {wrote} "
                    f"captured, <{min_frames}) — DROPPING")
                writer.abort()
                return
            write_episode_meta(
                writer, duration,
                success=(terminal == "completed"),
                note=(f"rollout {terminal}: {kept} frames kept of {wrote} captured, "
                      f"{n_takeover} takeover(s), {iv} human frames @ {FPS} Hz"),
                scene_tags=[],
                extra={"terminal": terminal, "rollout_id": self._rollout_id,
                       "n_takeovers": n_takeover, "human_frames": iv,
                       "used_throttle": bool(throttle),
                       "speed_factor": round(float(speed_max), 3),
                       **self._control_capability_meta()},
                # chunk-001 的 meta 落 episodes_stitched.jsonl (与离线 stitch 同一份
                # 契约, finalize_dagger_dataset.py 等下游按此读)。info.json 不动 —
                # 它由 episodes.jsonl 聚合, 而 chunk-001 不写那份; 离线 stitch 同样
                # 不碰 info.json, 训练集由 build_* 那步重建。
                filename="episodes_stitched.jsonl",
            )
            self.get_logger().info(
                f"  saved rollout ep={writer.ep} kept={kept}/{wrote} frames "
                f"({n_takeover} takeovers) duration={duration:.1f}s → {writer.root}")
        except Exception as e:
            self.get_logger().error(f"rollout finalize failed ({e}); aborting episode")
            try:
                writer.abort()
            except Exception:
                pass

    def _discard_rollout_episode(self, why: str) -> None:
        """Abort the WHOLE rollout episode. In direct-chunk001 mode a human segment
        cannot be discarded on its own — its frames are already interleaved into the
        one continuous episode — so 'discard' means dropping the whole attempt."""
        with self._lock:
            writer = self._roll_writer
            self._roll_writer = None
        if writer is None:
            return
        try:
            writer.abort()
            self.get_logger().warn(
                f"  DISCARDED whole rollout ep={writer.ep} ({why}) — chunk-001 是连续单集, "
                "无法只丢一段人接管")
        except Exception as e:
            self.get_logger().error(f"rollout discard abort failed: {e}")

    def _on_rollout_next(self, _msg) -> None:
        """Single-button rollout boundary (toggle). Only meaningful in POLICY_RUN.

        One "rollout" = one autonomous TASK ATTEMPT (a cloth fold, a pick-&-place,
        a wipe — whatever the policy is rolling out). NOT folding-specific.

        Press 1 (attempt complete): finalize the inference episode as a SUCCESS
          (terminal="completed") and PAUSE — execute=false so the operator can
          safely reset the scene. The policy model stays loaded/warm.
        Press 2 (scene reset done): START the next rollout — bump rollout_id, open
          a fresh inference episode, execute=true (the observe→execute transition
          flushes the policy's RTC action buffer, so no stale chunk carries over).

        Failures are NEVER marked here: an attempt that needed help was already cut
        by the takeover path (terminal="intervention", success=False). So the
        intervention-vs-success split is fully automatic — by HOW the episode
        ended, not by an operator choice. /dagger/rollout_paused is latch-published
        so the web UI can show which press (end vs start) comes next.
        """
        with self._lock:
            cur = self._state
            paused = self._rollout_paused
        if cur != State.POLICY_RUN:
            self.get_logger().warn(f"[rollout_next] ignored — state={cur.value} (only POLICY_RUN)")
            return
        if not paused:
            self.get_logger().info("[rollout_next] rollout complete → finalize episode (completed) + PAUSE for scene reset")
            if self._direct_c1:
                self._close_rollout_episode(terminal="completed")
            else:
                self._close_inference_episode(terminal="completed")
            self.pub_execute.publish(Bool(data=False))
            with self._lock:
                self._rollout_paused = True
        else:
            with self._lock:
                self._rollout_paused = False
                self._rollout_id += 1
                rid = self._rollout_id
            self.pub_execute.publish(Bool(data=True))  # flushes RTC on policy side
            self.get_logger().info(f"[rollout_next] START rollout_id={rid} (execute on, new inference ep next tick)")
        with self._lock:
            paused_now = self._rollout_paused
        self.pub_rollout_paused.publish(Bool(data=paused_now))

    # ── 30 Hz capture (Form C dual-writer dispatch) ──
    def _on_record_tick(self) -> None:
        """Capture one 30 Hz frame to whichever writer is active.

        Dual-writer routing (Form C, see docstring at top + §4.5):
          POLICY_RUN                 → inference_writer (intervention=0);
                                       lazy-opened on first tick with slave
                                       moved + RGB ready.
          HUMAN_RECORD + recording   → dagger writer (intervention=1);
                                       opened by pedal toggle.
          HUMAN_RECORD + !recording  → quiet (drag mode, no write).
          ALIGNING / RETURNING       → quiet (transition windows).

        Lazy-open guard: inference writer can't open during boot zero pose
        (would yield empty parquet) — wait for slave-moved and RGB frame
        before opening. Once opened, _close_inference_episode is the only
        path that resets the writer to None (called by _do_takeover
        step 1/4 or finalize()).
        """
        with self._lock:
            cur_state = self._state
            recording = self._recording
            dag_writer = self._writer
            inf_writer = self._inference_writer
            state = self._q_slave_left + self._q_slave_right
            # KAI0_ACTION_EQ_STATE=1 convention — official kai0_dagger format.
            # V3: 12 arm-joint action dims = slave state; 2 gripper dims (6=L,
            # 13=R) follow the master (teleop leader) grasp command. During
            # HUMAN_RECORD the master encoder publishes the human's intent; during
            # POLICY_RUN the master mirrors the slave so it ≈ state. Falls back to
            # slave gripper until a master JointState arrives.
            action = list(state)
            if self._grip_from_master:
                if self._got_master_left and (
                        cur_state == State.POLICY_RUN or self._master_available["L"]):
                    action[6] = self._q_master_left[6]
                if self._got_master_right and (
                        cur_state == State.POLICY_RUN or self._master_available["R"]):
                    action[13] = self._q_master_right[6]
            frames = {cam: self._rgb[cam] for cam in CAMERAS}
            depth_frames = {cam: self._depth.get(cam) for cam in DEPTH_CAMERAS}
            cur_speed = self._cur_speed_factor
            now = time.time()

        # 防黑帧: 任一相机还没出图就跳过本 tick — EpisodeWriter 对 None 帧填纯黑,
        # 会让 finalize 的 trim-validate 判黑帧 abort 整段 (dagger/inference 都护到)。
        if any(frames[cam] is None for cam in CAMERAS):
            return

        if self._direct_c1:
            self._tick_direct_chunk001(cur_state, recording, state, action,
                                       frames, cur_speed, now)
            return

        # ── HUMAN_RECORD + recording branch: write to dagger ──
        if cur_state == State.HUMAN_RECORD and recording and dag_writer is not None:
            try:
                dag_writer.write_tick(frames, state, action, now,
                                      depth_frames=depth_frames,
                                      intervention=1)
            except Exception as e:
                self.get_logger().error(
                    f"dagger write_tick failed (aborting recording): {e}"
                )
                with self._lock:
                    self._writer = None
                try:
                    dag_writer.abort()
                except Exception:
                    pass
                return
            with self._lock:
                self._wrote_frames += 1
                n = self._wrote_frames
            if n % (FPS * 5) == 0:
                self.get_logger().info(f"  dagger recording {n} frames ({n / FPS:.1f}s)")
            return

        # ── POLICY_RUN branch: write to inference (lazy-open if needed) ──
        if cur_state != State.POLICY_RUN:
            # ALIGNING / HUMAN_RECORD-not-recording / RETURNING — quiet
            return

        if not self._record_inference:
            # inference recording disabled by config — policy runs, nothing written
            return

        if inf_writer is None:
            # Paused between folds (button) — don't open a new inference ep yet.
            with self._lock:
                if self._rollout_paused:
                    return
            # Lazy-open: need slave moved + RGB before opening (else empty parquet).
            with self._lock:
                slave_ready = (any(abs(x) > 1e-4 for x in self._q_slave_left[:6]) and
                               any(abs(x) > 1e-4 for x in self._q_slave_right[:6]))
                # 必须【所有】相机都到过一帧再开 episode: _emit_tick 对 None 相机帧填黑,
                # 只等 top_head 会让手腕 D405 还没出图时首帧写成纯黑 → finalize 的
                # trim-validate 判黑帧 abort 整段。等齐 top_head+hand_left+hand_right。
                rgb_ready = all(self._rgb.get(cam) is not None for cam in CAMERAS)
            if not (slave_ready and rgb_ready):
                return
            self._open_inference_episode()
            with self._lock:
                inf_writer = self._inference_writer
            if inf_writer is None:
                return  # _open_inference_episode logged failure already

        # ── 单一 inference/ 数据集 + 整段加速标识 (最小改动方案): 所有 rollout 都写
        #    inference/; 本段 rollout 只要有一帧生效倍率 > 1.0 (踩过油门), 就把整段
        #    episode 的 meta 打上 used_throttle=true (整段标记, 与加速时长/占比无关),
        #    全程没踩则 false。下游按此标识区分"加速过 / 普通"数据, 不搬文件、不分目录。 ──
        if cur_speed > 1.0 + 1e-3 and not self._inf_throttle_used:
            with self._lock:
                self._inf_throttle_used = True
            self.get_logger().info(
                f"[SPEED] 本段 rollout 用过油门 (speed={cur_speed:.2f}) → meta used_throttle=true")

        try:
            inf_writer.write_tick(frames, state, action, now,
                                  depth_frames=depth_frames,
                                  intervention=0)
        except Exception as e:
            self.get_logger().error(
                f"inference write_tick failed (aborting): {e}"
            )
            with self._lock:
                self._inference_writer = None
            try:
                inf_writer.abort()
            except Exception:
                pass
            return
        with self._lock:
            self._inf_wrote_frames += 1
            n = self._inf_wrote_frames
            if cur_speed > self._inf_speed_max:
                self._inf_speed_max = cur_speed
        if n % (FPS * 10) == 0:
            self.get_logger().info(f"  inference recording {n} frames ({n / FPS:.1f}s)")
        return

    def _tick_direct_chunk001(self, cur_state: "State", recording: bool,
                              state: list, action: list, frames: dict,
                              cur_speed: float, now: float) -> None:
        """Single-writer capture for the whole rollout (KAI0_DIRECT_CHUNK001=1).

        Routing — identical set of KEPT frames to what the offline stitch produces:
          POLICY_RUN                → intervention=0
          HUMAN_RECORD + recording  → intervention=1
          ALIGNING / HUMAN_RECORD-not-recording / RETURNING / rollout paused
                                    → 静默 (离线 stitch 里这些帧本来就不存在)
        No depth: chunk-001 下游一帧不用 (dagger 采集里 depth 占 30G / RGB 只占 9.8G)。
        """
        writing_human = (cur_state == State.HUMAN_RECORD and recording)
        if not writing_human and cur_state != State.POLICY_RUN:
            return

        with self._lock:
            writer = self._roll_writer
            paused = self._rollout_paused

        if writer is None:
            if writing_human:
                # 不该发生 (rollout 必然起于 POLICY_RUN); 别静默吞掉人的纠错。
                self.get_logger().error(
                    "[direct-c1] HUMAN_RECORD 但本 rollout 无 writer — 该段接管丢失")
                return
            if paused:
                return   # 两次 rollout_next 之间, 操作员在复位场景
            # Lazy-open: 与旧路径同样的守卫 — 从臂离开 boot 零位 + 所有相机都出过图,
            # 否则首帧会是黑帧 / 空 parquet, finalize 的 trim-validate 会 abort 整段。
            with self._lock:
                slave_ready = (any(abs(x) > 1e-4 for x in self._q_slave_left[:6]) and
                               any(abs(x) > 1e-4 for x in self._q_slave_right[:6]))
            if not slave_ready:
                return
            self._open_rollout_episode()
            with self._lock:
                writer = self._roll_writer
            if writer is None:
                return  # _open_rollout_episode logged the failure

        if cur_speed > 1.0 + 1e-3 and not self._roll_throttle_used:
            with self._lock:
                self._roll_throttle_used = True
            self.get_logger().info(
                f"[SPEED] 本 rollout 用过油门 (speed={cur_speed:.2f}) → meta used_throttle=true")

        try:
            writer.write_tick(frames, state, action, now,
                              intervention=1 if writing_human else 0)
        except Exception as e:
            self.get_logger().error(f"rollout write_tick failed (aborting): {e}")
            with self._lock:
                self._roll_writer = None
            try:
                writer.abort()
            except Exception:
                pass
            return

        with self._lock:
            self._roll_frames += 1
            n = self._roll_frames
            if writing_human:
                self._roll_iv_frames += 1
            if cur_speed > self._roll_speed_max:
                self._roll_speed_max = cur_speed
        if n % (FPS * 10) == 0:
            self.get_logger().info(
                f"  rollout recording {n} frames ({n / FPS:.1f}s)")

    def finalize(self) -> None:
        """Best-effort cleanup on shutdown — close both writers if active."""
        with self._lock:
            cur = self._state
            dag_open = self._writer is not None
            inf_open = self._inference_writer is not None
            roll_open = self._roll_writer is not None
            was_recording = self._recording
        if roll_open:
            # 关栈时正卡在人接管段 → 先按段尾裁一次再收集, 否则遥操静止尾会留在集末。
            if was_recording:
                self._roll_segment("human_seg_end")
            self.get_logger().warn("shutdown during rollout recording — finalizing")
            self._close_rollout_episode(terminal="session_end")
        if not self._direct_c1 and (dag_open or cur == State.HUMAN_RECORD):
            self.get_logger().warn("shutdown during dagger recording — finalizing")
            self._close_episode()
        if inf_open:
            self.get_logger().warn("shutdown during inference recording — finalizing")
            self._close_inference_episode(terminal="session_end")


def main(args=None):
    rclpy.init(args=args)
    node = DaggerRecorder()
    # SIGTERM (ros2 launch / web 停止栈时发的信号, 非 Ctrl-C 的 SIGINT) 也要走
    # finalize → 存住正在录的 episode。抛 KeyboardInterrupt 让下面 finally 收尾。
    # (SIGKILL/-9 无法捕获 → 用 -9 杀 recorder 会丢未 finalize 的 episode。)
    import signal
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass  # 非主线程时无法注册, 忽略
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

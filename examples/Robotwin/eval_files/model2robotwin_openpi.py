from __future__ import annotations

"""openpi(pi05) ↔ RoboTwin eval 桥的 model client 适配器。

复用 robotwin_batch_bridge.py 的整套 env 驱动(slot/Pipe/success-rate/summary.json),
只把 model client 从 starVLA `ModelClient` 换成 openpi `WebsocketClientPolicy`。

与 starVLA 版(model2robotwin_interface.ModelClient)的关键差异:
  1. self.client = openpi_client.websocket_client_policy.WebsocketClientPolicy(host,port)
     (装在 kai0/.venv), 调 self.client.infer(obs) → {"actions": [T, D]} (单帧, 非 batch)。
  2. **不做客户端 unnormalize**: openpi server 端已在内部做 Normalize + AlohaOutputs(adapt_to_pi)
     反归一化, 返回的 actions 已是 RoboTwin joint 空间的真实动作 (14 维), 直接喂 env。
     (starVLA 版 server 发 normalized_actions, client 再 unnormalize — openpi 必须跳过, 否则双重反归一化。)
  3. **不校验 server metadata**: openpi metadata 无 ckpt_path 字段, 关掉校验。
  4. obs map: robotwin head/left/right_camera rgb → AlohaInputs 期望的
     images.{cam_high,cam_left_wrist,cam_right_wrist} (CHW uint8) + state[14](原始 joint) + prompt。
     server 内 AlohaInputs 做 _decode_state(adapt_to_pi) / ResizeImages→224, client 只发原始 obs。

env_action_type = "qpos" (14 维 joint), 与 starVLA joint 版一致。

batch_bridge 用到的方法/属性(签名与 starVLA ModelClient 一致):
  get_model / env_action_type / robotwin_mode / replan_steps / action_ensemble /
  action_ensemble_alpha / reset / needs_query / step_cached / build_example / step_batch / close。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# ⚠️ 不 import model2robotwin_interface: 它 top-level import starVLA(deployment/starVLA),
#   而 openpi bridge 跑在 RoboTwin conda env(无 starVLA)。故把用到的 _SlotState / joint
#   example 构造内联在此(纯 numpy, 与 starVLA 版语义一致)。


@dataclass
class _SlotState:
    """每个 env slot 的开环动作缓存 (与 model2robotwin_interface._SlotState 同语义)。"""

    task_description: str | None = None
    raw_actions: Optional[np.ndarray] = None
    action_cursor: int = 0
    executed_steps: int = 0

    def reset(self, task_description: str | None = None) -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.action_cursor = 0
        self.executed_steps = 0

    def needs_query(self) -> bool:
        return self.raw_actions is None or self.action_cursor >= int(self.raw_actions.shape[0])


def build_robotwin_example(task_description: str, observation: dict[str, Any]) -> dict[str, Any]:
    """joint 模式 example: {lang, image:[head,left,right], state[14]} (同 starVLA joint 版)。"""
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    state = observation.get("joint_action", {}).get("vector", None)
    example = {"lang": str(task_description), "image": [head_img, left_img, right_img], "state": state}
    for key in (
        "lmwm_transition_task",
        "lmwm_transition_current",
        "lmwm_transition_next",
        "lmwm_transition_mask",
        "lmwm_transition_history_images",
        "lmwm_transition_history_state",
    ):
        if key in observation:
            example[key] = observation[key]
    return example


def _to_chw_uint8(img: Any) -> np.ndarray:
    """RoboTwin rgb (H,W,3) → openpi AlohaInputs 期望的 (3,H,W) uint8。

    AlohaInputs._decode_aloha 内部对每张图做 rearrange('c h w -> h w c') 并把
    浮点图 *255→uint8, 故这里必须给 CHW。RoboTwin 原生给 uint8 HWC。"""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC/CHW image, got shape {arr.shape}")
    # 若已是 CHW (通道在前) 就原样; 否则 HWC→CHW。
    if arr.shape[0] == 3 and arr.shape[2] != 3:
        chw = arr
    else:
        chw = np.transpose(arr, (2, 0, 1))
    if np.issubdtype(chw.dtype, np.floating):
        chw = np.clip(chw * 255.0 if chw.max() <= 1.0 else chw, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(chw.astype(np.uint8))


class OpenpiRobotwinModelClient:
    """openpi pi05 RoboTwin eval client。接口与 starVLA ModelClient 平行 (batch_bridge 无感知)。"""

    _DEFAULT_HOST = "127.0.0.1"
    _DEFAULT_PORT = 8000

    def __init__(
        self,
        policy_ckpt_path: Optional[str] = None,
        *,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        replan_steps: Optional[int | str] = None,
        action_ensemble: Optional[bool | str] = None,
        action_ensemble_alpha: Optional[float | str] = None,
        hint_encoder: Optional[str] = None,
        hint_device: str = "cuda",
        **_ignored: Any,
    ) -> None:
        # 控制契约: joint 14 维, env_action_type=qpos (与 starVLA joint 版一致)。
        self.policy_ckpt_path = str(policy_ckpt_path) if policy_ckpt_path else None
        self.robotwin_mode = "joint"
        self.env_action_type = "qpos"
        self.replan_steps = self._resolve_replan_steps(replan_steps)
        # openpi pi05 chunk 短 (action_horizon=8), 默认执行整段后再 requery。
        self.action_ensemble = self._as_bool(action_ensemble)
        self.action_ensemble_alpha = float(action_ensemble_alpha) if action_ensemble_alpha not in (None, "") else 0.0
        if self.action_ensemble:
            raise NotImplementedError(
                "OpenpiRobotwinModelClient 暂不支持 action_ensemble (openpi chunk 已短, 直接 replan)。"
            )

        from openpi_client import websocket_client_policy  # kai0/.venv

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))
        # metadata bypass: openpi metadata 无 ckpt_path, 不做校验, 仅打印。
        try:
            meta = self.client.get_server_metadata() or {}
        except Exception:
            meta = {}

        # a1: 在线 hint. hint_encoder 非空 → 加载 robotwin HintComputer, 每帧对 head_camera 算 hint。
        self.hint_computer = None
        self.hint_encoder = hint_encoder
        self.hint_device = hint_device
        self._slot_states: dict[int, _SlotState] = {}

        print(
            "*** OpenpiRobotwinModelClient "
            f"host={host} port={port} robotwin_mode={self.robotwin_mode} "
            f"env_action_type={self.env_action_type} replan_steps={self.replan_steps} "
            f"hint_encoder={hint_encoder} server_meta_keys={sorted(meta.keys())} ***",
            flush=True,
        )

    def _ensure_hint_computer(self) -> None:
        if self.hint_computer is None and self.hint_encoder:
            from examples.Robotwin.eval_files.hint_online_robotwin import RobotwinHintComputer

            self.hint_computer = RobotwinHintComputer(
                encoder=self.hint_encoder,
                device=self.hint_device,
            )

    # ---- slot 缓存 (复用 _SlotState 语义) ----
    def _get_slot_state(self, slot_id: int) -> _SlotState:
        slot_id = int(slot_id)
        if slot_id not in self._slot_states:
            self._slot_states[slot_id] = _SlotState()
        return self._slot_states[slot_id]

    def reset(self, task_description: str = "", slot_id: int = 0) -> None:
        self._get_slot_state(slot_id).reset(task_description=task_description)

    def reset_slots(self, slot_ids=None) -> None:
        if slot_ids is None:
            self._slot_states.clear()
            return
        for slot_id in slot_ids:
            self._slot_states.pop(int(slot_id), None)

    def needs_query(self, slot_id: int = 0, task_description: Optional[str] = None) -> bool:
        slot_state = self._get_slot_state(slot_id)
        if task_description is not None:
            resolved = str(task_description)
            if resolved != slot_state.task_description:
                slot_state.reset(task_description=resolved)
        return slot_state.needs_query()

    def step_cached(self, slot_id: int = 0, task_description: Optional[str] = None) -> np.ndarray:
        if self.needs_query(slot_id=slot_id, task_description=task_description):
            raise RuntimeError(f"Slot {slot_id} has no cached actions available.")
        return self._pop_slot_action(slot_id)

    def _pop_slot_action(self, slot_id: int) -> np.ndarray:
        slot_state = self._get_slot_state(int(slot_id))
        if slot_state.raw_actions is None or slot_state.action_cursor >= int(slot_state.raw_actions.shape[0]):
            raise RuntimeError(f"Slot {slot_id} has no cached actions after inference.")
        action = np.asarray(slot_state.raw_actions[slot_state.action_cursor], dtype=np.float32)
        slot_state.action_cursor += 1
        slot_state.executed_steps += 1
        return action

    # ---- example 构造 (与 starVLA joint 版一致: {lang, image:[head,left,right], state[14]}) ----
    def build_example(self, task_description: str, observation: dict[str, Any]) -> dict[str, Any]:
        return build_robotwin_example(task_description, observation)

    def step(self, example: dict[str, Any], step: int = 0, slot_id: int = 0) -> np.ndarray:
        del step
        return self.step_batch([example], slot_ids=[slot_id])[0]

    def _build_openpi_obs(self, example: dict[str, Any]) -> dict[str, Any]:
        images = list(example["image"])  # [head, left, right] HWC uint8
        if len(images) < 3:
            raise ValueError(f"expected 3 robotwin images (head,left,right), got {len(images)}")
        head, left, right = images[0], images[1], images[2]
        obs: dict[str, Any] = {
            "images": {
                "cam_high": _to_chw_uint8(head),
                "cam_left_wrist": _to_chw_uint8(left),
                "cam_right_wrist": _to_chw_uint8(right),
            },
            "prompt": str(example.get("lang", "")),
        }
        state = example.get("state", None)
        if state is not None:
            obs["state"] = np.asarray(state, dtype=np.float32).reshape(-1)
        for key in (
            "lmwm_transition_task",
            "lmwm_transition_current",
            "lmwm_transition_next",
            "lmwm_transition_mask",
            "lmwm_transition_history_images",
            "lmwm_transition_history_state",
        ):
            if key in example:
                if key == "lmwm_transition_history_images":
                    obs[key] = np.stack([_to_chw_uint8(image) for image in example[key]])
                else:
                    obs[key] = np.asarray(example[key])
        # a1: 在线 hint 塞入 obs.lmwm_hint (AlohaInputs 透传 → model)。head 是原始 RoboTwin rgb HWC。
        self._ensure_hint_computer()
        if self.hint_computer is not None:
            obs["lmwm_hint"] = self.hint_computer.compute(np.asarray(head))[None].astype(np.float32)
        return obs

    def step_batch(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        slot_ids: Optional[Sequence[int]] = None,
    ) -> list[np.ndarray]:
        if slot_ids is None:
            slot_ids = list(range(len(examples)))
        if len(examples) != len(slot_ids):
            raise ValueError(f"examples/slot_ids length mismatch: {len(examples)} vs {len(slot_ids)}")

        for example, slot_id in zip(examples, slot_ids):
            slot_id = int(slot_id)
            slot_state = self._get_slot_state(slot_id)
            task_description = str(example.get("lang", ""))
            if task_description != slot_state.task_description:
                slot_state.reset(task_description=task_description)
            if not slot_state.needs_query():
                continue
            # openpi server 单帧 infer → {"actions": [T, 14]} (已反归一化的真实 joint 动作)。
            obs = self._build_openpi_obs(example)
            result = self.client.infer(obs)
            if "actions" not in result:
                raise KeyError(f"openpi server response missing 'actions': keys={list(result.keys())}")
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            if actions.ndim != 2:
                raise ValueError(f"expected actions [T,D], got shape {actions.shape}")
            if self.replan_steps is not None:
                actions = actions[: int(self.replan_steps)]
            if int(actions.shape[0]) <= 0:
                raise RuntimeError("openpi server returned empty action chunk.")
            slot_state.raw_actions = actions
            slot_state.action_cursor = 0

        return [np.asarray(self._pop_slot_action(int(slot_id)), dtype=np.float32) for slot_id in slot_ids]

    def close(self) -> None:
        try:
            ws = getattr(self.client, "_ws", None)
            if ws is not None:
                ws.close()
        except Exception:
            pass

    @staticmethod
    def _resolve_replan_steps(replan_steps: Optional[int | str]) -> Optional[int]:
        if replan_steps is None or replan_steps == "":
            return None
        resolved = int(str(replan_steps).strip())
        return resolved if resolved > 0 else None

    @staticmethod
    def _as_bool(value: Optional[bool | str]) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_model(usr_args):
    """robotwin_batch_bridge.py 通过 ROBOTWIN_MODEL_INTERFACE=openpi 路由到此处。"""
    hint_encoder = os.getenv("ROBOTWIN_HINT_ENCODER") or None
    if os.getenv("OPENPI_SERVER_HINT_ENCODER"):
        hint_encoder = None
    return OpenpiRobotwinModelClient(
        policy_ckpt_path=usr_args.get("policy_ckpt_path"),
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 8000),
        replan_steps=usr_args.get("replan_steps", os.getenv("ROBOTWIN_REPLAN_STEPS")),
        action_ensemble=usr_args.get("action_ensemble", os.getenv("ROBOTWIN_ACTION_ENSEMBLE")),
        action_ensemble_alpha=usr_args.get("action_ensemble_alpha", os.getenv("ROBOTWIN_ACTION_ENSEMBLE_ALPHA")),
        hint_encoder=hint_encoder,
        hint_device=os.getenv("ROBOTWIN_HINT_DEVICE", "cuda"),
    )


def reset_model(model):
    model.reset_slots()

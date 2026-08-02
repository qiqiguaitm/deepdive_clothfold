"""RoboTwin official-evaluator adapter for an OpenPI websocket policy server."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _chw_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC image, got {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        chw = array
    else:
        chw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(chw, dtype=np.uint8)


class OfficialOpenPiModel:
    def __init__(self) -> None:
        self.client = WebsocketClientPolicy(
            host=os.environ.get("OPENPI_WS_HOST", "127.0.0.1"),
            port=int(os.environ.get("OPENPI_WS_PORT", "11000")),
        )
        self.pi0_step = int(os.environ.get("ROBOTWIN_REPLAN_STEPS", "50"))


def get_model(_usr_args: dict[str, Any]) -> OfficialOpenPiModel:
    return OfficialOpenPiModel()


def eval(task_env: Any, model: OfficialOpenPiModel, observation: dict[str, Any]) -> None:
    camera = observation["observation"]
    result = model.client.infer(
        {
            "images": {
                "cam_high": _chw_uint8(camera["head_camera"]["rgb"]),
                "cam_left_wrist": _chw_uint8(camera["left_camera"]["rgb"]),
                "cam_right_wrist": _chw_uint8(camera["right_camera"]["rgb"]),
            },
            "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
            "prompt": str(task_env.get_instruction()),
        }
    )
    actions = np.asarray(result["actions"], dtype=np.float32)[: model.pi0_step, :14]
    for action in actions:
        task_env.take_action(action)


def reset_model(_model: OfficialOpenPiModel) -> None:
    return None

"""GR00T N1.7-compatible bimanual end-effector features.

Each arm pose is expressed in that arm's own ``base_link`` frame.  The Piper
DH model's link6 origin equals ``gripper_base`` in the deployed URDF, so it is
the stable tool frame used here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


JOINT_DIM = 14
EEF_DIM = 9
STATE_DIM = JOINT_DIM + 2 * EEF_DIM
STATE_EEF_LEFT = slice(14, 23)
STATE_EEF_RIGHT = slice(23, 32)
ACTION_EEF_LEFT = STATE_EEF_LEFT
ACTION_EEF_RIGHT = STATE_EEF_RIGHT

# Piper SDK C_PiperForwardKinematics(dh_is_offset=0x01), in metres.
_DH_A = np.array([0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0])
_DH_ALPHA = np.array([0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2])
_DH_THETA = np.array(
    [0.0, -math.pi * 172.22 / 180.0, -math.pi * 102.78 / 180.0, 0.0, 0.0, 0.0]
)
_DH_D = np.array([0.123, 0.0, 0.0, 0.25075, 0.0, 0.091])


def _link_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def piper_fk_matrix(joints: Sequence[float]) -> np.ndarray:
    """Return ``T_base_link6`` for six Piper joint angles in radians."""
    q = np.asarray(joints, dtype=np.float64)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("Piper FK requires six finite joint angles")
    transform = np.eye(4, dtype=np.float64)
    for i in range(6):
        transform = transform @ _link_transform(
            float(_DH_ALPHA[i]),
            float(_DH_A[i]),
            float(q[i] + _DH_THETA[i]),
            float(_DH_D[i]),
        )
    return transform


def matrix_to_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    """Encode the first two rotation-matrix columns in GR00T's 6D form."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must have shape (3, 3)")
    return matrix[:, :2].reshape(6)


def rotation_6d_to_matrix(rotation_6d: Sequence[float]) -> np.ndarray:
    """Decode continuous rotation 6D with Gram-Schmidt orthogonalisation."""
    columns = np.asarray(rotation_6d, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-12:
        raise ValueError("rotation 6D first axis has zero norm")
    first = first / first_norm
    second = columns[:, 1] - np.dot(first, columns[:, 1]) * first
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-12:
        raise ValueError("rotation 6D axes are collinear")
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def _eef_9d(joints: Sequence[float]) -> list[float]:
    transform = piper_fk_matrix(joints)
    return np.concatenate(
        (transform[:3, 3], matrix_to_rotation_6d(transform[:3, :3]))
    ).astype(np.float32).tolist()


def append_absolute_eef(joint_state: Sequence[float]) -> list[float]:
    """Append left/right absolute EEF poses to a legacy 14-D joint vector."""
    joints = [float(value) for value in joint_state]
    if len(joints) != JOINT_DIM or not np.all(np.isfinite(joints)):
        raise ValueError("EEF capture requires a finite 14-D bimanual joint state")
    return joints + _eef_9d(joints[0:6]) + _eef_9d(joints[7:13])


def _relative_eef(current: Sequence[float], following: Sequence[float]) -> list[float]:
    current = np.asarray(current, dtype=np.float64)
    following = np.asarray(following, dtype=np.float64)
    translation_delta = following[:3] - current[:3]
    # Base-frame delta: R_next = R_delta @ R_current.
    rotation_delta = (
        rotation_6d_to_matrix(following[3:])
        @ rotation_6d_to_matrix(current[3:]).T
    )
    return np.concatenate(
        (translation_delta, matrix_to_rotation_6d(rotation_delta))
    ).astype(np.float32).tolist()


def apply_relative_eef_actions(
    states: Sequence[Sequence[float]],
    legacy_actions: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Build 32-D actions after trimming, using the next *kept* state pose."""
    if len(states) != len(legacy_actions):
        raise ValueError("state/action row counts differ")
    actions: list[list[float]] = []
    identity_6d = matrix_to_rotation_6d(np.eye(3)).astype(np.float32).tolist()
    terminal_delta = [0.0, 0.0, 0.0] + identity_6d
    for index, (state, legacy_action) in enumerate(zip(states, legacy_actions)):
        state_row = list(state)
        action_row = [float(value) for value in list(legacy_action)[:JOINT_DIM]]
        if len(state_row) != STATE_DIM or len(action_row) != JOINT_DIM:
            raise ValueError("expected 32-D state and 14-D legacy action rows")
        if index + 1 < len(states):
            next_state = states[index + 1]
            left = _relative_eef(
                state_row[STATE_EEF_LEFT], next_state[STATE_EEF_LEFT]
            )
            right = _relative_eef(
                state_row[STATE_EEF_RIGHT], next_state[STATE_EEF_RIGHT]
            )
        else:
            left, right = list(terminal_delta), list(terminal_delta)
        actions.append(action_row + left + right)
    return actions


def modality_config() -> dict:
    fields = {
        "left_joint_position": {"start": 0, "end": 6},
        "left_gripper_position": {"start": 6, "end": 7},
        "right_joint_position": {"start": 7, "end": 13},
        "right_gripper_position": {"start": 13, "end": 14},
        "left_eef_9d": {"start": 14, "end": 23},
        "right_eef_9d": {"start": 23, "end": 32},
    }
    cameras = ("top_head", "mid_head", "hand_left", "hand_right")
    return {
        "state": dict(fields),
        "action": dict(fields),
        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in cameras
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }


def write_modality_json(dataset_root: Path) -> Path:
    path = Path(dataset_root) / "meta" / "modality.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(modality_config(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path

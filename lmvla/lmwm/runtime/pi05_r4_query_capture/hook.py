"""Capture the observations actually queried by a RoboTwin policy rollout."""

from __future__ import annotations

import builtins
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


_TARGET_MODULE = "envs._base_task"
_PATCHED = False


def _query_path(ffmpeg: Any) -> Path:
    args = getattr(ffmpeg, "args", ())
    if not args:
        raise RuntimeError("R4 query capture cannot resolve the evaluation video path")
    video = Path(str(args[-1]))
    return video.with_name(f"query_{video.stem}.npz")


def _reset(task: Any, ffmpeg: Any) -> None:
    task._r4_query_capture = {
        "path": _query_path(ffmpeg),
        "frame_index": [],
        "states": [],
        "cam_high": [],
        "cam_left_wrist": [],
        "cam_right_wrist": [],
        "instruction": None,
    }


def _record(task: Any, observation: dict[str, Any]) -> None:
    capture = getattr(task, "_r4_query_capture", None)
    if capture is None:
        return
    frame = int(task.take_action_cnt)
    if frame % 50 != 0:
        return
    if capture["frame_index"] and int(capture["frame_index"][-1]) == frame:
        return
    images = observation["observation"]
    state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    cameras = {
        "cam_high": np.asarray(images["head_camera"]["rgb"]),
        "cam_left_wrist": np.asarray(images["left_camera"]["rgb"]),
        "cam_right_wrist": np.asarray(images["right_camera"]["rgb"]),
    }
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(f"invalid R4 query state: shape={state.shape}")
    for name, image in cameras.items():
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"invalid R4 query image {name}: shape={image.shape}, dtype={image.dtype}"
            )
    capture["frame_index"].append(frame)
    capture["states"].append(state.copy())
    for name, image in cameras.items():
        capture[name].append(np.ascontiguousarray(image))
    capture["instruction"] = str(task.get_instruction())


def _finalize(task: Any) -> None:
    capture = getattr(task, "_r4_query_capture", None)
    if capture is None:
        return
    task._r4_query_capture = None
    frames = np.asarray(capture["frame_index"], dtype=np.int64)
    if len(frames) == 0:
        raise ValueError("R4 query capture contains no policy queries")
    if frames[0] != 0 or np.any(np.diff(frames) <= 0):
        raise ValueError(f"invalid R4 query frame sequence: {frames.tolist()}")
    destination = Path(capture["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        query_frame_index=frames,
        query_states=np.stack(capture["states"]).astype(np.float32, copy=False),
        cam_high=np.stack(capture["cam_high"]),
        cam_left_wrist=np.stack(capture["cam_left_wrist"]),
        cam_right_wrist=np.stack(capture["cam_right_wrist"]),
        instruction=np.asarray(str(capture["instruction"])),
    )
    temporary.replace(destination)


def patch_base_task(module: Any) -> None:
    global _PATCHED
    if _PATCHED:
        return
    cls = module.Base_Task
    original_set_video = cls._set_eval_video_ffmpeg
    original_get_obs = cls.get_obs
    original_del_video = cls._del_eval_video_ffmpeg

    def set_video(self: Any, ffmpeg: Any) -> Any:
        result = original_set_video(self, ffmpeg)
        _reset(self, ffmpeg)
        return result

    def get_obs(self: Any) -> dict[str, Any]:
        observation = original_get_obs(self)
        _record(self, observation)
        return observation

    def del_video(self: Any) -> Any:
        _finalize(self)
        return original_del_video(self)

    cls._set_eval_video_ffmpeg = set_video
    cls.get_obs = get_obs
    cls._del_eval_video_ffmpeg = del_video
    _PATCHED = True


def _try_patch_loaded_module() -> bool:
    module = sys.modules.get(_TARGET_MODULE)
    if module is None or not hasattr(module, "Base_Task"):
        return False
    patch_base_task(module)
    return True


def install() -> None:
    """Patch Base_Task after RoboTwin imports it, without changing frozen sources."""
    if os.environ.get("R4_CAPTURE_QUERY_OBSERVATIONS") != "1" or _try_patch_loaded_module():
        return
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def restore_if_patched() -> None:
        if _try_patch_loaded_module():
            builtins.__import__ = original_import
            importlib.import_module = original_import_module

    def importing(name: str, *args: Any, **kwargs: Any) -> Any:
        result = original_import(name, *args, **kwargs)
        restore_if_patched()
        return result

    def importing_module(name: str, package: str | None = None) -> Any:
        result = original_import_module(name, package)
        restore_if_patched()
        return result

    builtins.__import__ = importing
    importlib.import_module = importing_module

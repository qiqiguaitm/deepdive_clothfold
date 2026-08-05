from __future__ import annotations

from types import SimpleNamespace
import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from hook import patch_base_task


class FakeBaseTask:
    def __init__(self):
        self.take_action_cnt = 0
        self.instruction = "stack the blocks"
        self.eval_video_ffmpeg = None

    def _set_eval_video_ffmpeg(self, ffmpeg):
        self.eval_video_ffmpeg = ffmpeg

    def _del_eval_video_ffmpeg(self):
        self.eval_video_ffmpeg = None

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        image = np.full((4, 6, 3), self.take_action_cnt, dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image + 1},
                "right_camera": {"rgb": image + 2},
            },
            "joint_action": {"vector": np.arange(14, dtype=np.float32)},
        }


def test_query_capture_is_atomic_and_deduplicates_frames(tmp_path):
    import hook

    hook._PATCHED = False
    patch_base_task(SimpleNamespace(Base_Task=FakeBaseTask))
    video = tmp_path / "episode7.mp4"
    ffmpeg = SimpleNamespace(args=["ffmpeg", str(video)])
    task = FakeBaseTask()
    task._set_eval_video_ffmpeg(ffmpeg)
    task.get_obs()
    task.get_obs()
    task.take_action_cnt = 50
    task.get_obs()
    task.take_action_cnt = 73
    task.get_obs()
    task._del_eval_video_ffmpeg()

    output = tmp_path / "query_episode7.npz"
    assert output.is_file()
    with np.load(output, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["query_frame_index"], [0, 50])
        assert payload["query_states"].shape == (2, 14)
        assert payload["cam_high"].shape == (2, 4, 6, 3)
        assert payload["cam_left_wrist"].dtype == np.uint8
        assert payload["instruction"].item() == "stack the blocks"
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_install_patches_dynamic_importlib_task_load(monkeypatch):
    import hook

    class DynamicBaseTask(FakeBaseTask):
        pass

    module = SimpleNamespace(Base_Task=DynamicBaseTask)
    original_import_module = importlib.import_module

    def dynamic_import(name, package=None):
        if name == hook._TARGET_MODULE:
            monkeypatch.setitem(sys.modules, name, module)
            return module
        return original_import_module(name, package)

    monkeypatch.delitem(sys.modules, hook._TARGET_MODULE, raising=False)
    monkeypatch.setenv("R4_CAPTURE_QUERY_OBSERVATIONS", "1")
    monkeypatch.setattr(importlib, "import_module", dynamic_import)
    hook._PATCHED = False
    hook.install()

    imported = importlib.import_module(hook._TARGET_MODULE)

    assert imported is module
    assert hook._PATCHED
    assert DynamicBaseTask.get_obs is not FakeBaseTask.get_obs
    assert importlib.import_module is dynamic_import


def test_sitecustomize_installs_hook_in_fresh_spawn_interpreter():
    hook_dir = Path(__file__).parent
    environment = os.environ.copy()
    environment["R4_CAPTURE_QUERY_OBSERVATIONS"] = "1"
    environment["PYTHONPATH"] = str(hook_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sitecustomize,hook; print(sitecustomize.__file__); "
            "print(hook._PATCHED); print(hook._TARGET_MODULE)",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().splitlines()
    assert Path(lines[0]).resolve() == (hook_dir / "sitecustomize.py").resolve()
    assert lines[1:] == ["False", "envs._base_task"]

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_mt3_tracker_data.py")
SPEC = importlib.util.spec_from_file_location("audit_mt3_tracker_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_history_frames_clip_at_episode_start():
    assert MODULE.history_frames(4) == (0, 0, 4)
    assert MODULE.history_frames(20) == (5, 13, 20)


def test_episode_paths_follow_lerobot_chunk_layout(tmp_path):
    path = MODULE.episode_path(
        tmp_path,
        "videos",
        24750,
        "mp4",
        "observation.images.cam_high",
    )
    assert path == tmp_path / "videos/chunk-024/observation.images.cam_high/episode_024750.mp4"

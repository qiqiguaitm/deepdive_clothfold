from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_mt_stage_tracker_split.py")
SPEC = importlib.util.spec_from_file_location("build_mt_stage_tracker_split", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_preserves_singletons_and_supported_stage_coverage():
    episode_stages = {
        0: {0, 3},
        1: {0, 1},
        2: {0, 1},
        3: {0, 2},
        4: {0, 2},
        5: {0},
    }
    train, val, _ = MODULE.split_task_episodes(episode_stages, val_count=2, seed=7)
    assert 0 in train
    assert not set(train).intersection(val)
    for stage in (0, 1, 2):
        members = {ep for ep, stages in episode_stages.items() if stage in stages}
        assert members.intersection(train)
        assert members.intersection(val)

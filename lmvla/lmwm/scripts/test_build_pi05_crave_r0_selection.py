from __future__ import annotations

import numpy as np

from build_pi05_crave_r0_selection import (
    EPISODES_PER_TASK,
    TASK_BLOCKS,
    build_selection,
)


def test_selection_is_disjoint_and_covers_six_task_panel(tmp_path) -> None:
    train = []
    heldout = []
    panel_episodes = []
    for block in TASK_BLOCKS.values():
        lower = block * EPISODES_PER_TASK
        task_heldout = list(range(lower, lower + 10))
        task_train = list(range(lower + 10, lower + 230))
        heldout.extend(task_heldout)
        train.extend(task_train)
        panel_episodes.extend(task_heldout)
    panel = {
        "cur_ep": np.asarray(panel_episodes, dtype=np.int32),
        "cur_fi": np.zeros(len(panel_episodes), dtype=np.int32),
    }
    manifest, selected_rows, required = build_selection(
        {"heldout_episodes": heldout, "train_episodes": train}, panel, tmp_path
    )
    assert manifest["task_count"] == 6
    assert manifest["reference_episode_count"] == 1200
    assert manifest["heldout_episode_count"] == 60
    assert len(selected_rows) == 60
    assert len(required) == 1260
    for row in manifest["tasks"].values():
        assert not set(row["reference_episodes"]) & set(row["heldout_episodes"])

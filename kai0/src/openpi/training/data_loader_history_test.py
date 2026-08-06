import numpy as np
import pytest

from openpi.training import data_loader
from openpi.training import config


class _Table:
    def __init__(self, rows):
        self._rows = rows
        self.hf_dataset = {
            "episode_index": [row["episode_index"] for row in rows],
            "frame_index": [row["frame_index"] for row in rows],
        }

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def _row(episode, frame):
    return {
        "episode_index": episode,
        "frame_index": frame,
        "observation.images.cam_high": np.full((3, 2, 2), frame, dtype=np.uint8),
        "observation.state": np.arange(14, dtype=np.float32) + frame,
    }


def test_same_episode_history_clamps_at_start_and_preserves_episode():
    dataset = _Table([*[_row(1, frame) for frame in range(4)], *[_row(2, frame) for frame in range(4)]])
    wrapped = data_loader.SameEpisodeHistoryDataset(dataset, (-3, -1, 0))

    item = wrapped[5]  # episode 2, frame 1

    np.testing.assert_array_equal(item["lmwm_transition_history_images"][:, 0, 0, 0], [0, 0, 1])
    np.testing.assert_array_equal(item["lmwm_transition_history_state"][:, 0], [0, 0, 1])
    assert item["episode_index"] == 2


def test_same_episode_history_rejects_invalid_offsets_and_missing_frames():
    dataset = _Table([_row(1, 0), _row(1, 2)])
    with pytest.raises(ValueError, match="end at zero"):
        data_loader.SameEpisodeHistoryDataset(dataset, (-1,))

    wrapped = data_loader.SameEpisodeHistoryDataset(dataset, (-1, 0))
    with pytest.raises(KeyError, match="history frame absent"):
        wrapped[1]


def test_mt3_inference_configs_register_both_frozen_tracker_candidates():
    for candidate in ("current_frame", "history_proprio"):
        value = config.get_config(f"pi05_robotwin_mt3_learned_{candidate}_exact")
        assert value.model.lmwm_transition_condition == "learned"
        assert value.model.lmwm_transition_tracker == candidate

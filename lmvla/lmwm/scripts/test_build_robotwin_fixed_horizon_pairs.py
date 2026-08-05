import json

import numpy as np

from build_robotwin_fixed_horizon_pairs import build_pairs


def test_fixed_horizon_pairs_are_exact_and_do_not_clamp(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"episode_index": 3, "length": 5},
                {"episode_index": 4, "length": 2},
            )
        )
        + "\n"
    )
    pairs = build_pairs(episodes, fps=2, horizon_seconds=1.0)
    np.testing.assert_array_equal(pairs["cur_ep"], [3, 3, 3])
    np.testing.assert_array_equal(pairs["cur_fi"], [0, 1, 2])
    np.testing.assert_array_equal(pairs["tgt_fi"], [2, 3, 4])
    assert int(pairs["horizon_frames"]) == 2


def test_fixed_horizon_requires_exact_frame_offset(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text('{"episode_index": 0, "length": 10}\n')
    try:
        build_pairs(episodes, fps=2, horizon_seconds=0.75)
    except ValueError as error:
        assert "exactly" in str(error)
    else:
        raise AssertionError("non-integral horizon should fail")

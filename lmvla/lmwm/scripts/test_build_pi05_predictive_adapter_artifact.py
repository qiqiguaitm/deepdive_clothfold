import json

import numpy as np

from build_pi05_predictive_adapter_artifact import build_artifact
from build_pi05_predictive_adapter_artifact import select_heldout_eval_rows


def test_predictive_artifact_is_episode_disjoint_and_never_clamps(tmp_path):
    dataset = tmp_path / "data"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"fps": 2, "total_frames": 10, "total_tasks": 2})
    )
    (meta / "tasks.jsonl").write_text(
        '{"task_index": 0, "task": "a"}\n{"task_index": 1, "task": "b"}\n'
    )
    (meta / "episodes.jsonl").write_text(
        '{"episode_index": 0, "tasks": ["a"], "length": 5}\n'
        '{"episode_index": 1, "tasks": ["b"], "length": 5}\n'
    )
    pairs = tmp_path / "pairs.npz"
    np.savez_compressed(
        pairs,
        cur_ep=np.asarray([0, 0, 0, 1, 1, 1]),
        cur_fi=np.asarray([0, 1, 2, 0, 1, 2]),
        tgt_fi=np.asarray([2, 3, 4, 2, 3, 4]),
        horizon_frames=np.asarray(2),
        fps=np.asarray(2),
        horizon_seconds=np.asarray(1.0),
    )

    arrays, metadata, manifest = build_artifact(
        dataset, pairs, split_seed=3, heldout_fraction=0.5, action_horizon=3
    )

    assert np.all(arrays["tgt_fi"] == arrays["cur_fi"] + 2)
    assert np.all(arrays["target_valid"])
    assert metadata["audit"]["split"]["leakage"] is False
    assert {row["split"] for row in manifest} == {"train", "heldout"}
    assert arrays["action_padding"].tolist() == [0, 0, 0, 0, 0, 0]

    eval_rows = select_heldout_eval_rows(arrays, sample_count=3, seed=7)
    heldout_episode = int(arrays["cur_ep"][arrays["heldout"]][0])
    assert eval_rows["row_index"].shape == (3,)
    assert np.all(eval_rows["heldout"])
    assert set(eval_rows["cur_ep"]) == {heldout_episode}


def test_eval_panel_covers_every_heldout_episode():
    arrays = {
        "cur_ep": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "heldout": np.asarray([False, False, True, True, True, True]),
        "cur_fi": np.arange(6, dtype=np.int32),
    }
    first = select_heldout_eval_rows(arrays, sample_count=3, seed=11)
    second = select_heldout_eval_rows(arrays, sample_count=3, seed=11)
    assert set(first["cur_ep"]) == {1, 2}
    assert np.array_equal(first["row_index"], second["row_index"])

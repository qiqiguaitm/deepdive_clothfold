from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from build_pi05_r4_training_chunks import build, outcome_weights
from audit_pi05_r4_query_dataset import REQUIRED_TASKS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_outcome_weights_are_task_normalized_and_have_exp_ratio() -> None:
    tasks = np.asarray(["a", "a", "b", "b", "b"])
    success = np.asarray([False, True, False, True, True])
    weights = outcome_weights(tasks, success, 1.0)
    for task in ("a", "b"):
        assert weights[tasks == task].mean() == pytest.approx(1.0)
    assert weights[1] / weights[0] == pytest.approx(np.e)
    assert weights[3] / weights[2] == pytest.approx(np.e)


def test_builds_fifty_step_chunks_only_after_full_audit(tmp_path: Path) -> None:
    query_records = []
    outcome_records = []
    policy = "a" * 64
    for task in sorted(REQUIRED_TASKS):
        for success_index, success in enumerate((False, True)):
            seed = 1000 + success_index
            trajectory = tmp_path / f"{task}_{seed}_trajectory.npz"
            query = tmp_path / f"{task}_{seed}_query.npz"
            actions = np.arange(75 * 14, dtype=np.float32).reshape(75, 14)
            states = actions / 100.0
            np.savez_compressed(
                trajectory,
                actions=actions,
                states=states,
                frame_index=np.arange(75),
            )
            frames = np.asarray([0, 50])
            cameras = np.zeros((2, 4, 5, 3), dtype=np.uint8)
            np.savez_compressed(
                query,
                query_frame_index=frames,
                query_states=states[frames],
                cam_high=cameras,
                cam_left_wrist=cameras,
                cam_right_wrist=cameras,
                instruction=np.asarray(f"do {task}"),
            )
            record = {
                "task": task,
                "scene_seed": seed,
                "success": success,
                "split": "train",
                "behavior_policy_sha256": policy,
                "trajectory": trajectory.name,
                "trajectory_sha256": digest(trajectory),
                "query_observations": query.name,
                "query_observations_sha256": digest(query),
                "video": "unused.mp4",
                "video_sha256": "unused",
            }
            query_records.append(record)
            outcome_records.append(dict(record))
    query_manifest = tmp_path / "query_manifest.json"
    outcome_manifest = tmp_path / "outcome_manifest.json"
    query_manifest.write_text(
        json.dumps(
            {
                "protocol": "pi05_r4_policy_query_observations_combined_v1",
                "behavior_policy_sha256": policy,
                "records": query_records,
            }
        )
    )
    outcome_manifest.write_text(
        json.dumps({"behavior_policy_sha256": policy, "records": outcome_records})
    )
    output = tmp_path / "chunks.npz"

    report = build(query_manifest, outcome_manifest, output)

    assert report["record_count"] == 12
    assert report["sample_count"] == 24
    assert report["interpretation"].endswith("world-critic estimate")
    with np.load(output, allow_pickle=False) as payload:
        assert payload["action"].shape == (24, 50, 14)
        assert payload["state"].shape == (24, 14)
        assert payload["action_valid"].shape == (24, 50)
        assert payload["action_valid"][:, :25].all()
        second_queries = payload["query_frame"] == 50
        assert payload["action_valid"][second_queries].sum(axis=1).tolist() == [25] * 12
        assert np.allclose(
            payload["outcome_calibrated_weight"].reshape(6, 4).mean(axis=1), 1.0
        )

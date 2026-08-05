from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from build_pi05_r4_crave_weight_sidecar import (
    normalized_progress_weights,
    validate_outcome_free_manifest,
    verify_chunk_alignment,
)
from build_pi05_r4_outcome_free_manifest import FORBIDDEN_KEYS, project


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_projection_removes_every_outcome_and_trajectory_field(tmp_path: Path) -> None:
    query = tmp_path / "query.npz"
    np.savez(query, query_frame_index=np.asarray([0]))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "protocol": "pi05_r4_policy_query_observations_combined_v1",
                "behavior_policy_sha256": "a" * 64,
                "records": [
                    {
                        "behavior_policy_sha256": "a" * 64,
                        "episode_id": 0,
                        "eval_seed": 0,
                        "query_observations": query.name,
                        "query_observations_sha256": digest(query),
                        "scene_seed": 3,
                        "source_manifest_index": 0,
                        "split": "train",
                        "success": True,
                        "task": "task_a",
                        "trajectory": "trajectory.npz",
                        "video": "episode.mp4",
                    }
                ],
            }
        )
    )
    result = project(source)
    assert result["record_count"] == 1
    assert not (FORBIDDEN_KEYS & set(result["records"][0]))
    assert "trajectory" not in result["records"][0]
    assert "video" not in result["records"][0]
    assert {"success", "trajectory", "video"} <= set(result["omitted_record_fields"])


def test_outcome_free_validator_rejects_leaked_success(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "pi05_r4_outcome_free_query_inputs_v1",
                "record_count": 1,
                "records": [{"task": "a", "success": False}],
            }
        )
    )
    with pytest.raises(ValueError, match="leaked"):
        validate_outcome_free_manifest(path)


def test_crave_weights_are_positive_task_normalized_and_terminal_neutral() -> None:
    tasks = np.asarray(["a", "a", "a", "b", "b"])
    delta = np.asarray([-0.2, 0.4, 0.0, 0.1, 0.0], dtype=np.float32)
    mask = np.asarray([True, True, False, True, False])
    weights = normalized_progress_weights(tasks, delta, mask, temperature=1.0)
    assert np.all(weights > 0)
    assert np.isclose(weights[tasks == "a"].mean(), 1.0)
    assert np.isclose(weights[tasks == "b"].mean(), 1.0)
    assert weights[1] > weights[0]
    # The unlabeled terminal sample starts from raw weight one; normalization is task-wide.
    expected_b = np.asarray([np.exp(0.0), 1.0])
    expected_b /= expected_b.mean()
    assert np.allclose(weights[tasks == "b"], expected_b)


def test_crave_weights_reject_task_without_labeled_transition() -> None:
    with pytest.raises(ValueError, match="no CRAVE-labeled"):
        normalized_progress_weights(
            np.asarray(["a"]),
            np.asarray([0.0]),
            np.asarray([False]),
            temperature=1.0,
        )


def test_sidecar_must_exactly_align_with_action_chunks(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.npz"
    np.savez(
        chunks,
        task=np.asarray(["a", "a"]),
        scene_seed=np.asarray([1, 1]),
        query_index=np.asarray([0, 1]),
        query_frame=np.asarray([0, 50]),
    )
    sidecar = {
        "task": np.asarray(["a", "a"]),
        "scene_seed": np.asarray([1, 1]),
        "query_index": np.asarray([0, 1]),
        "query_frame": np.asarray([0, 50]),
    }
    verify_chunk_alignment(sidecar, chunks)
    sidecar["query_frame"] = np.asarray([0, 51])
    with pytest.raises(ValueError, match="query_frame"):
        verify_chunk_alignment(sidecar, chunks)

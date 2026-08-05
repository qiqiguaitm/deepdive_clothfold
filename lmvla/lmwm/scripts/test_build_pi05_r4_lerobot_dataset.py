from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from build_pi05_r4_lerobot_dataset import build, feature_spec, validate_arrays


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_spec_uses_direct_fifty_step_action_chunks() -> None:
    features = feature_spec((24, 32, 3))
    assert features["action"]["shape"] == (50, 14)
    assert features["observation.images.cam_high"]["dtype"] == "image"
    assert features["sample_weight"]["shape"] == (1,)


def test_validate_arrays_rejects_task_without_both_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        state=np.zeros((1, 14), np.float32),
        action=np.zeros((1, 50, 14), np.float32),
        action_valid=np.ones((1, 50), bool),
        task=np.asarray(["only"]),
        scene_seed=np.asarray([1]),
        success=np.asarray([True]),
        record_index=np.asarray([0]),
        query_index=np.asarray([0]),
        query_frame=np.asarray([0]),
        query_observations=np.asarray(["query.npz"]),
        instruction=np.asarray(["do only"]),
        ordinary_weight=np.ones(1, np.float32),
        outcome_calibrated_weight=np.ones(1, np.float32),
    )
    with np.load(path, allow_pickle=False) as payload, pytest.raises(
        ValueError, match="lacks success/failure"
    ):
        validate_arrays(payload)


def test_builds_loadable_direct_chunk_dataset(tmp_path: Path) -> None:
    pytest.importorskip("lerobot")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:  # pragma: no cover - compatibility with pre-0.4 layouts.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    query = tmp_path / "query.npz"
    images = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    states = np.arange(28, dtype=np.float32).reshape(2, 14)
    np.savez_compressed(
        query,
        query_frame_index=np.asarray([0, 50]),
        query_states=states,
        cam_high=images,
        cam_left_wrist=images,
        cam_right_wrist=images,
        instruction=np.asarray("do task"),
    )
    chunks = tmp_path / "chunks.npz"
    actions = np.zeros((2, 50, 14), dtype=np.float32)
    np.savez_compressed(
        chunks,
        state=states,
        action=actions,
        action_valid=np.ones((2, 50), bool),
        task=np.asarray(["task", "task"]),
        task_id=np.asarray([0, 0]),
        scene_seed=np.asarray([1, 2]),
        success=np.asarray([False, True]),
        record_index=np.asarray([0, 1]),
        query_index=np.asarray([0, 1]),
        query_frame=np.asarray([0, 50]),
        query_observations=np.asarray([query.name, query.name]),
        instruction=np.asarray(["do task", "do task"]),
        ordinary_weight=np.ones(2, np.float32),
        outcome_calibrated_weight=np.ones(2, np.float32),
    )
    chunks_report = tmp_path / "chunks_report.json"
    chunks_report.write_text(
        json.dumps(
            {
                "chunks_sha256": digest(chunks),
                "source_audit": {"accepted": True},
                "record_count": 2,
                "sample_count": 2,
            }
        )
    )
    output = tmp_path / "dataset"

    report = build(
        chunks,
        chunks_report,
        output,
        repo_id="local/r4-test",
        image_writer_threads=1,
    )

    assert report["episodes"] == report["samples"] == 2
    assert report["action_sequence_keys"] == []
    dataset = LeRobotDataset(
        "local/r4-test", root=output, delta_timestamps={}, video_backend="pyav"
    )
    assert len(dataset) == 2
    sample = dataset[0]
    assert tuple(sample["action"].shape) == (50, 14)
    assert tuple(sample["observation.images.cam_high"].shape) == (3, 24, 32)
    assert sample["sample_weight"].ndim == 0

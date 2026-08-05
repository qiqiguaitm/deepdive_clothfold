from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).with_name("train_mt3_stage_tracker.py")
SPEC = importlib.util.spec_from_file_location("train_mt3_stage_tracker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_class_weights_are_finite_normalized_and_clipped():
    balanced = np.repeat(np.arange(10), 3)
    np.testing.assert_allclose(MODULE.class_weights(balanced), np.ones(10))
    imbalanced = np.concatenate([np.zeros(1000, dtype=int), np.repeat(np.arange(1, 10), 1)])
    weights = MODULE.class_weights(imbalanced)
    assert np.all(np.isfinite(weights))
    assert np.min(weights) >= 0.25
    assert np.max(weights) <= 4.0
    assert weights[0] < weights[-1]


def test_structurally_absent_head_class_gets_zero_weight():
    current = MODULE.class_weights(np.repeat(np.arange(9), 2))
    next_stage = MODULE.class_weights(np.repeat(np.arange(1, 10), 2))
    assert current[9] == 0.0
    assert next_stage[0] == 0.0
    np.testing.assert_allclose(current[:9], np.ones(9))
    np.testing.assert_allclose(next_stage[1:], np.ones(9))


def test_tracker_shapes_match_frozen_protocol():
    current = MODULE.CurrentFrameTracker()
    history = MODULE.HistoryProprioTracker()
    for model, features in (
        (current, torch.zeros(2, 3, 2048)),
        (history, torch.zeros(2, 3, 2062)),
    ):
        current_logits, next_logits = model(features)
        assert current_logits.shape == (2, 10)
        assert next_logits.shape == (2, 10)


def test_feature_loader_checks_shards_and_builds_candidate_inputs(tmp_path: Path):
    root = tmp_path / "features"
    shard = root / "shard-00-of-01"
    shard.mkdir(parents=True)
    chunk = shard / "features-00000.npz"
    np.savez(
        chunk,
        episode=np.asarray([1, 2]),
        frame=np.asarray([3, 4]),
        task=np.asarray([0, 1]),
        current_target=np.asarray([0, 1]),
        next_target=np.asarray([1, 2]),
        split=np.asarray([0, 1]),
        current_view_features=np.zeros((2, 3, 2048), dtype=np.float16),
        history_base_features=np.zeros((2, 3, 2048), dtype=np.float16),
        history_proprio=np.zeros((2, 3, 14), dtype=np.float32),
    )
    digest = MODULE.sha256(chunk)
    (shard / "manifest.json").write_text(
        json.dumps(
            {
                "num_shards": 1,
                "shard_index": 0,
                "rows": 2,
                "provenance": {"split_sha256": "frozen"},
                "chunks": [{"file": chunk.name, "rows": 2, "sha256": digest}],
            }
        )
    )
    current, _ = MODULE.load_features(root, "current_frame")
    history, _ = MODULE.load_features(root, "history_proprio")
    assert current["features"].shape == (2, 3, 2048)
    assert history["features"].shape == (2, 3, 2062)

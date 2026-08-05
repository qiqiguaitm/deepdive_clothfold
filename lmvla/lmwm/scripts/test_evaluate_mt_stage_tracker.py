from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("evaluate_mt_stage_tracker.py")
SPEC = importlib.util.spec_from_file_location("evaluate_mt_stage_tracker", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def logits(labels: list[int], classes: int = 4) -> np.ndarray:
    result = np.full((len(labels), classes), -4.0, dtype=np.float32)
    result[np.arange(len(labels)), labels] = 4.0
    return result


def test_metrics_are_task_stratified_and_switch_delay_is_stable_prediction_delay():
    current = np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])
    prediction = np.asarray([0, 0, 0, 1, 1, 0, 0, 1, 1, 1])
    data = {
        "episode": np.asarray([10] * 5 + [11] * 5),
        "frame": np.asarray([0, 1, 2, 3, 4] * 2),
        "task": np.asarray([0] * 5 + [1] * 5),
        "current_target": current,
        "next_target": current + 1,
        "current_logits": logits(prediction),
        "next_logits": logits((current + 1).tolist()),
    }
    result = MODULE.evaluate(data)
    assert set(result["tasks"]) == {"0", "1"}
    assert result["pooled"]["next_accuracy"] == 1.0
    assert result["pooled"]["switch_delay"]["true_switches"] == 2
    assert result["pooled"]["switch_delay"]["detected_switches"] == 1
    assert result["pooled"]["switch_delay"]["mean_frames"] == 0.0


def test_duplicate_rows_are_rejected():
    data = {
        "episode": np.asarray([10, 10]),
        "frame": np.asarray([0, 0]),
        "task": np.asarray([0, 0]),
        "current_target": np.asarray([0, 0]),
        "next_target": np.asarray([1, 1]),
        "current_logits": logits([0, 0]),
        "next_logits": logits([1, 1]),
    }
    try:
        MODULE.evaluate(data)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate rows should fail")


def test_training_episode_is_rejected_by_frozen_split():
    data = {
        "episode": np.asarray([10]),
        "frame": np.asarray([0]),
        "task": np.asarray([0]),
        "current_target": np.asarray([0]),
        "next_target": np.asarray([1]),
        "current_logits": logits([0]),
        "next_logits": logits([1]),
    }
    try:
        MODULE.evaluate(data, split={"train_episodes": [10], "val_episodes": [11]})
    except ValueError as error:
        assert "training episodes" in str(error)
    else:
        raise AssertionError("training episode should fail")

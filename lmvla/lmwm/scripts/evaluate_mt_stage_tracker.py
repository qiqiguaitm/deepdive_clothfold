#!/usr/bin/env python3
"""Evaluate frozen-split milestone stage-tracker predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = (
    "episode",
    "frame",
    "task",
    "current_target",
    "next_target",
    "current_logits",
    "next_logits",
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in np.unique(target):
        true_positive = np.sum((target == label) & (prediction == label))
        false_positive = np.sum((target != label) & (prediction == label))
        false_negative = np.sum((target == label) & (prediction != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def expected_calibration_error(probability: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    prediction = np.argmax(probability, axis=-1)
    confidence = np.max(probability, axis=-1)
    correct = prediction == target
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(selected):
            error += np.mean(selected) * abs(np.mean(correct[selected]) - np.mean(confidence[selected]))
    return float(error)


def switch_delays(
    episode: np.ndarray,
    frame: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    min_stable_frames: int = 3,
) -> dict[str, Any]:
    delays = []
    total = 0
    for episode_id in np.unique(episode):
        selected = np.flatnonzero(episode == episode_id)
        selected = selected[np.argsort(frame[selected], kind="stable")]
        true_sequence = target[selected]
        pred_sequence = prediction[selected]
        boundaries = np.flatnonzero(true_sequence[1:] != true_sequence[:-1]) + 1
        for boundary_index, boundary in enumerate(boundaries):
            total += 1
            stop = int(boundaries[boundary_index + 1]) if boundary_index + 1 < len(boundaries) else len(selected)
            new_stage = true_sequence[boundary]
            first_stable = None
            for candidate in range(int(boundary), stop - min_stable_frames + 1):
                if np.all(pred_sequence[candidate : candidate + min_stable_frames] == new_stage):
                    first_stable = candidate
                    break
            if first_stable is not None:
                delays.append(int(frame[selected[first_stable]] - frame[selected[boundary]]))
    values = np.asarray(delays, dtype=np.float64)
    return {
        "true_switches": total,
        "detected_switches": len(delays),
        "detection_rate": 1.0 if total == 0 else len(delays) / total,
        "mean_frames": None if not delays else float(np.mean(values)),
        "median_frames": None if not delays else float(np.median(values)),
        "p90_frames": None if not delays else float(np.percentile(values, 90)),
        "min_stable_frames": min_stable_frames,
    }


def metrics(data: dict[str, np.ndarray], selected: np.ndarray) -> dict[str, Any]:
    current_target = data["current_target"][selected]
    next_target = data["next_target"][selected]
    current_probability = softmax(data["current_logits"][selected])
    next_probability = softmax(data["next_logits"][selected])
    current_prediction = np.argmax(current_probability, axis=-1)
    next_prediction = np.argmax(next_probability, axis=-1)
    return {
        "rows": int(np.sum(selected)),
        "current_macro_f1": macro_f1(current_target, current_prediction),
        "current_accuracy": float(np.mean(current_prediction == current_target)),
        "next_accuracy": float(np.mean(next_prediction == next_target)),
        "current_ece_15bin": expected_calibration_error(current_probability, current_target),
        "next_ece_15bin": expected_calibration_error(next_probability, next_target),
        "switch_delay": switch_delays(
            data["episode"][selected],
            data["frame"][selected],
            current_target,
            current_prediction,
        ),
    }


def validate_split(data: dict[str, np.ndarray], split: dict[str, Any]) -> None:
    train = {int(value) for value in split["train_episodes"]}
    validation = {int(value) for value in split["val_episodes"]}
    observed = {int(value) for value in data["episode"]}
    leaked = observed.intersection(train)
    if leaked:
        raise ValueError(f"prediction archive contains {len(leaked)} training episodes")
    unknown = observed.difference(validation)
    if unknown:
        raise ValueError(f"prediction archive contains {len(unknown)} episodes outside the frozen validation split")


def evaluate(
    data: dict[str, np.ndarray],
    *,
    split: dict[str, Any] | None = None,
    split_sha256: str | None = None,
) -> dict[str, Any]:
    missing = set(REQUIRED_KEYS).difference(data)
    if missing:
        raise ValueError(f"prediction archive is missing keys: {sorted(missing)}")
    row_count = len(data["episode"])
    if any(len(data[key]) != row_count for key in REQUIRED_KEYS):
        raise ValueError("prediction arrays do not have a common row count")
    if data["current_logits"].ndim != 2 or data["next_logits"].ndim != 2:
        raise ValueError("current_logits and next_logits must be rank-2")
    if len(set(zip(data["episode"].tolist(), data["frame"].tolist(), strict=True))) != row_count:
        raise ValueError("duplicate episode/frame rows in prediction archive")
    if split is not None:
        validate_split(data, split)

    all_rows = np.ones(row_count, dtype=bool)
    tasks = {
        str(int(task)): metrics(data, data["task"] == task)
        for task in sorted(np.unique(data["task"]))
    }
    task_macro = {
        key: float(np.mean([float(value[key]) for value in tasks.values()]))
        for key in ("current_macro_f1", "current_accuracy", "next_accuracy")
    }
    return {
        "protocol": {
            "split_unit": "episode",
            "split_sha256": split_sha256,
            "ece_bins": 15,
            "switch_delay_definition": (
                "frames from a true stage boundary to the first three consecutive predictions "
                "of the new stage, censored at the next true boundary"
            ),
        },
        "pooled": metrics(data, all_rows),
        "task_macro": task_macro,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = np.load(args.predictions)
    split = json.loads(args.split_manifest.read_text())
    result = evaluate(
        {key: archive[key] for key in archive.files},
        split=split,
        split_sha256=hashlib.sha256(args.split_manifest.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

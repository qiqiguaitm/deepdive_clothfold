#!/usr/bin/env python3
"""Fit frozen CRAVE R0 readouts and apply the preregistered paired gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


CONDITIONS = ("normal", "shuffled", "masked", "current", "time")
TARGETS = ("progress_change", "target_recurrence_density", "phase_boundary_crossing")
METRICS = {
    "progress_change": "absolute_error",
    "target_recurrence_density": "absolute_error",
    "phase_boundary_crossing": "brier_score",
}
PROJECTION_DIM = 256
PROJECTION_SEED = 20260804
RIDGE_ALPHA = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_features(
    train: np.ndarray, evaluations: dict[str, np.ndarray], *, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train = np.asarray(train, dtype=np.float32)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-5, 1.0, std)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((train.shape[1], PROJECTION_DIM), dtype=np.float32)
    projection /= np.sqrt(train.shape[1])
    projected_train = ((train - mean) / std) @ projection
    projected_mean = projected_train.mean(axis=0)
    projected_std = np.where(projected_train.std(axis=0) < 1e-5, 1.0, projected_train.std(axis=0))
    projected_train = (projected_train - projected_mean) / projected_std
    projected_eval = {
        name: (((np.asarray(value, dtype=np.float32) - mean) / std) @ projection - projected_mean)
        / projected_std
        for name, value in evaluations.items()
    }
    return projected_train, projected_eval


def time_features(labels: dict[str, np.ndarray], task_count: int) -> np.ndarray:
    time = np.asarray(labels["normalized_current_time"], dtype=np.float32)
    task = np.asarray(labels["physical_task"], dtype=np.int64)
    if np.any((task < 0) | (task >= task_count)):
        raise ValueError("physical task id outside frozen vocabulary")
    powers = np.stack([time**degree for degree in range(6)], axis=1)
    one_hot = np.eye(task_count, dtype=np.float32)[task]
    return (one_hot[:, :, None] * powers[:, None, :]).reshape(len(time), -1)


def fit_readout(train_x: np.ndarray, train_y: np.ndarray):
    target_mean = train_y.mean(axis=0)
    target_std = np.where(train_y.std(axis=0) < 1e-6, 1.0, train_y.std(axis=0))
    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(train_x, (train_y - target_mean) / target_std)
    return model, target_mean, target_std


def predict_readout(model, target_mean, target_std, values: np.ndarray) -> np.ndarray:
    prediction = model.predict(values) * target_std + target_mean
    prediction[:, 2] = np.clip(prediction[:, 2], 0.0, 1.0)
    return prediction.astype(np.float32)


def row_losses(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    losses = np.empty_like(prediction, dtype=np.float32)
    losses[:, :2] = np.abs(prediction[:, :2] - target[:, :2])
    losses[:, 2] = np.square(prediction[:, 2] - target[:, 2])
    return losses


def paired_episode_bootstrap(
    episodes: np.ndarray,
    normal_loss: np.ndarray,
    control_loss: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    unique, inverse = np.unique(episodes, return_inverse=True)
    counts = np.bincount(inverse)
    differences = control_loss - normal_loss
    episode_means = np.bincount(inverse, weights=differences) / counts
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 512):
        stop = min(start + 512, draws)
        sampled = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        estimates[start:stop] = episode_means[sampled].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "episode_count": int(len(unique)),
        "mean_control_loss_minus_normal": float(episode_means.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--eval-labels", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--features-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()

    train_labels = load_npz(args.train_labels)
    eval_labels = load_npz(args.eval_labels)
    features = load_npz(args.features)
    labels_manifest = json.loads(args.labels_manifest.read_text())
    features_manifest = json.loads(args.features_manifest.read_text())
    task_count = len(labels_manifest["physical_task_ids"])
    if features_manifest["future_image_used"]:
        raise ValueError("R0 probe features may not consume future images")
    if not np.array_equal(features["train_row"], np.arange(len(train_labels["cur_ep"]))):
        raise ValueError("train feature rows do not align with frozen labels")
    if not np.array_equal(features["eval_row"], np.arange(len(eval_labels["cur_ep"]))):
        raise ValueError("eval feature rows do not align with frozen labels")
    if set(map(int, train_labels["cur_ep"])) & set(map(int, eval_labels["cur_ep"])):
        raise ValueError("probe train/eval episode leakage")

    train_y = np.stack(
        [np.asarray(train_labels[name], dtype=np.float32) for name in TARGETS], axis=1
    )
    eval_y = np.stack(
        [np.asarray(eval_labels[name], dtype=np.float32) for name in TARGETS], axis=1
    )
    normal_train, action_eval = project_features(
        features["train_normal"],
        {name: features[f"eval_{name}"] for name in ("normal", "shuffled", "masked")},
        seed=PROJECTION_SEED,
    )
    action_model = fit_readout(normal_train, train_y)
    predictions = {
        name: predict_readout(*action_model, values)
        for name, values in action_eval.items()
    }
    current_train, current_eval = project_features(
        features["train_current"],
        {"current": features["eval_current"]},
        seed=PROJECTION_SEED,
    )
    current_model = fit_readout(current_train, train_y)
    predictions["current"] = predict_readout(
        *current_model, current_eval["current"]
    )
    time_train = time_features(train_labels, task_count)
    time_eval = time_features(eval_labels, task_count)
    time_model = fit_readout(time_train, train_y)
    predictions["time"] = predict_readout(*time_model, time_eval)

    losses = {name: row_losses(eval_y, prediction) for name, prediction in predictions.items()}
    episodes = np.asarray(eval_labels["cur_ep"], dtype=np.int64)
    comparisons = {}
    for control_index, control in enumerate(CONDITIONS[1:], start=1):
        comparisons[control] = {}
        for target_index, target in enumerate(TARGETS):
            comparisons[control][target] = paired_episode_bootstrap(
                episodes,
                losses["normal"][:, target_index],
                losses[control][:, target_index],
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed + 10 * control_index + target_index,
            )
    progress_pass = all(
        comparisons[control]["progress_change"]["ci95_low"] > 0.0
        for control in CONDITIONS[1:]
    )
    density_pass = all(
        comparisons[control]["target_recurrence_density"]["ci95_low"] > 0.0
        for control in CONDITIONS[1:]
    )
    boundary_pass = all(
        comparisons[control]["phase_boundary_crossing"]["ci95_low"] > 0.0
        for control in CONDITIONS[1:]
    )
    accepted = bool(progress_pass and (density_pass or boundary_pass))

    task_results = {}
    for task, task_id in labels_manifest["physical_task_ids"].items():
        rows = np.flatnonzero(eval_labels["physical_task"] == task_id)
        task_results[task] = {
            condition: {
                target: float(losses[condition][rows, index].mean())
                for index, target in enumerate(TARGETS)
            }
            for condition in CONDITIONS
        }
    report = {
        "schema_version": 1,
        "protocol": "pi05_crave_r0_control_semantics_gate_v1",
        "projection_dim": PROJECTION_DIM,
        "projection_seed": PROJECTION_SEED,
        "ridge_alpha": RIDGE_ALPHA,
        "readout_protocol": {
            "normal_shuffled_masked": "one readout fit on normal train features and reused unchanged for action interventions",
            "current": "independent matched-capacity readout",
            "time": "independent task-specific degree-5 polynomial ridge readout",
        },
        "targets": dict(zip(TARGETS, (METRICS[name] for name in TARGETS), strict=True)),
        "train_rows": int(len(train_y)),
        "eval_rows": int(len(eval_y)),
        "eval_episodes": int(len(np.unique(episodes))),
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
        "aggregate_loss": {
            condition: {
                target: float(losses[condition][:, index].mean())
                for index, target in enumerate(TARGETS)
            }
            for condition in CONDITIONS
        },
        "comparisons": comparisons,
        "task_results": task_results,
        "gate_components": {
            "progress_beats_all_controls": progress_pass,
            "density_beats_all_controls": density_pass,
            "boundary_beats_all_controls": boundary_pass,
        },
        "accepted": accepted,
        "train_labels_sha256": sha256(args.train_labels),
        "eval_labels_sha256": sha256(args.eval_labels),
        "features_sha256": sha256(args.features),
        "labels_manifest_sha256": sha256(args.labels_manifest),
        "features_manifest_sha256": sha256(args.features_manifest),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    verdict = "accepted" if accepted else "rejected"
    marker = args.gate_dir / f"r0_gate.{verdict}"
    other = args.gate_dir / f"r0_gate.{'rejected' if accepted else 'accepted'}"
    other.unlink(missing_ok=True)
    marker_tmp = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    marker_tmp.write_text(json.dumps({"accepted": accepted}) + "\n")
    marker_tmp.replace(marker)
    print(json.dumps({"gate_components": report["gate_components"], "accepted": accepted}, indent=2))


if __name__ == "__main__":
    main()

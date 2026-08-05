#!/usr/bin/env python3
"""Package and validate the six-task causal recurrence readout for pi0.5 R2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from build_pi05_crave_r0_labels import query_recurrence_fields, reference_ranges
from pi05_r2_adaptive_execution import CausalExecutionController, R2ScheduleConfig


VALIDATION_SEED = 20260804
DEFAULT_VALIDATION_PAIRS_PER_TASK = 128
MAX_FIELD_MAE = 5e-4
MIN_BOUNDARY_AUC = 0.55
LOW_DENSITY_QUANTILE = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    lower = 0
    while lower < len(scores):
        upper = lower + 1
        while upper < len(scores) and sorted_scores[upper] == sorted_scores[lower]:
            upper += 1
        ranks[order[lower:upper]] = 0.5 * (lower + 1 + upper)
        lower = upper
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def phase_boundary_anchors(
    progress: np.ndarray, phase: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.int32)
    boundary = np.asarray(boundary, dtype=bool)
    anchors = [
        float(np.median(progress[boundary & (phase == phase_id)]))
        for phase_id in sorted(set(phase[boundary].tolist()))
        if np.any(boundary & (phase == phase_id))
    ]
    return np.asarray(sorted(set(anchors)), dtype=np.float32)


def boundary_proximity(progress: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=np.float32)
    output = np.zeros(len(progress), dtype=np.float32)
    for index, value in enumerate(progress):
        ahead = anchors[anchors >= value]
        if len(ahead):
            output[index] = np.exp(-float(ahead[0] - value) / 0.05)
    return output


def stable_sample(rows: np.ndarray, count: int, *, task: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) <= count:
        return rows
    keys = [
        hashlib.sha256(f"{VALIDATION_SEED}:{task}:{int(row)}".encode()).digest()
        for row in rows
    ]
    order = sorted(range(len(rows)), key=keys.__getitem__)
    return rows[np.asarray(order[:count], dtype=np.int64)]


def validate_streaming_events(
    *, labels: dict[str, np.ndarray], rows: np.ndarray
) -> dict[str, int]:
    identities = sorted(
        {(int(labels["cur_ep"][row]), int(labels["physical_task"][row])) for row in rows}
    )
    stall = 0
    regression = 0
    observations = 0
    for episode, task_id in identities:
        episode_rows = rows[
            (labels["cur_ep"][rows] == episode)
            & (labels["physical_task"][rows] == task_id)
        ]
        episode_rows = episode_rows[np.argsort(labels["cur_fi"][episode_rows])]
        controller = CausalExecutionController(R2ScheduleConfig())
        for row in episode_rows:
            confidence = float(
                np.mean(
                    labels["current_recurrence_density"][rows]
                    <= labels["current_recurrence_density"][row]
                )
            )
            signal = controller.observe(
                step=int(labels["cur_fi"][row]),
                progress=float(labels["current_progress"][row]),
                density=float(labels["current_recurrence_density"][row]),
                confidence=confidence,
                boundary_proximity=0.0,
            )
            stall += int(signal.stall)
            regression += int(signal.regression)
            observations += 1
    return {
        "episodes": len(identities),
        "observations": observations,
        "stall_observations": stall,
        "regression_observations": regression,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--probe-labels", type=Path, required=True)
    parser.add_argument("--reference-trajectories", type=Path, required=True)
    parser.add_argument("--reference-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--validation-pairs-per-task", type=int, default=DEFAULT_VALIDATION_PAIRS_PER_TASK
    )
    args = parser.parse_args()
    if args.validation_pairs_per_task <= 0:
        raise ValueError("validation pair count must be positive")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    labels_manifest = json.loads(args.labels_manifest.read_text(encoding="utf-8"))
    labels_npz = np.load(args.labels)
    labels = {name: labels_npz[name] for name in labels_npz.files}
    probe_npz = np.load(args.probe_labels)
    probe = {name: probe_npz[name] for name in probe_npz.files}
    reference_npz = np.load(args.reference_trajectories)
    reference = {name: reference_npz[name] for name in reference_npz.files}
    task_ids = {str(task): int(value) for task, value in labels_manifest["physical_task_ids"].items()}
    if sorted(task_ids) != sorted(selection["tasks"]):
        raise ValueError("selection and label task sets differ")

    packed: dict[str, np.ndarray] = {
        "task_names": np.asarray(sorted(task_ids), dtype=np.str_),
    }
    reports: dict[str, Any] = {}
    accepted = True
    all_validation_rows: list[np.ndarray] = []
    for task in sorted(task_ids):
        task_id = task_ids[task]
        reference_episodes = list(map(int, selection["tasks"][task]["reference_episodes"]))
        episode_features = [
            np.asarray(
                np.load(args.reference_feature_dir / f"ep{episode}.npz")["pooled"],
                dtype=np.float32,
            )
            for episode in reference_episodes
        ]
        features, ranges = reference_ranges(episode_features)
        offsets = np.asarray([ranges[0][0], *[upper for _, upper in ranges]], dtype=np.int64)
        reference_rows = np.flatnonzero(reference["physical_task"] == task_id)
        expected_episode = np.concatenate(
            [np.full(len(values), episode, dtype=np.int32) for episode, values in zip(reference_episodes, episode_features, strict=True)]
        )
        expected_frame = np.concatenate(
            [np.arange(len(values), dtype=np.int32) for values in episode_features]
        )
        if not np.array_equal(reference["episode"][reference_rows], expected_episode):
            raise ValueError(f"{task}: reference episode identity mismatch")
        if not np.array_equal(reference["frame"][reference_rows], expected_frame):
            raise ValueError(f"{task}: reference frame identity mismatch")
        reference_progress = np.asarray(reference["progress"][reference_rows], dtype=np.float32)
        reference_density = np.asarray(reference["recurrence_density"][reference_rows], dtype=np.float32)
        anchors = phase_boundary_anchors(
            reference_progress,
            reference["phase"][reference_rows],
            reference["phase_boundary"][reference_rows],
        )
        calibration = np.asarray(
            probe["current_recurrence_density"][probe["physical_task"] == task_id],
            dtype=np.float32,
        )
        prefix = f"task{task_id}_"
        packed[prefix + "features"] = features.astype(np.float16)
        packed[prefix + "episode_offsets"] = offsets.astype(np.int32)
        packed[prefix + "progress"] = reference_progress
        packed[prefix + "density"] = reference_density
        packed[prefix + "density_calibration"] = np.sort(calibration)
        packed[prefix + "boundary_progress"] = anchors
        packed[prefix + "sigma"] = np.asarray(labels_manifest["tasks"][task]["sigma"], dtype=np.float32)

        task_rows = np.flatnonzero(labels["physical_task"] == task_id)
        validation_rows = stable_sample(
            task_rows, args.validation_pairs_per_task, task=task
        )
        all_validation_rows.append(validation_rows)
        query_features = np.stack(
            [
                np.asarray(
                    np.load(args.reference_feature_dir / f"ep{int(labels['cur_ep'][row])}.npz")["pooled"][
                        int(labels["cur_fi"][row])
                    ],
                    dtype=np.float32,
                )
                for row in validation_rows
            ]
        )
        predicted_density, predicted_progress = query_recurrence_fields(
            query_features,
            features,
            ranges,
            float(labels_manifest["tasks"][task]["sigma"]),
            device=args.device,
            chunk_size=args.chunk_size,
        )
        progress_mae = float(
            np.mean(np.abs(predicted_progress - labels["current_progress"][validation_rows]))
        )
        density_mae = float(
            np.mean(
                np.abs(predicted_density - labels["current_recurrence_density"][validation_rows])
            )
        )
        proximity = boundary_proximity(predicted_progress, anchors)
        boundary_labels = np.asarray(labels["phase_boundary_crossing"][validation_rows], dtype=bool)
        auc = roc_auc(boundary_labels, proximity)
        auc_estimable = int(boundary_labels.sum()) >= 5 and int((~boundary_labels).sum()) >= 5
        task_accept = bool(
            progress_mae <= MAX_FIELD_MAE
            and density_mae <= MAX_FIELD_MAE
            and auc_estimable
            and auc is not None
            and auc >= MIN_BOUNDARY_AUC
        )
        accepted = accepted and task_accept
        reports[task] = {
            "task_id": task_id,
            "reference_episodes": len(reference_episodes),
            "reference_rows": len(features),
            "validation_rows": len(validation_rows),
            "progress_mae": progress_mae,
            "density_mae": density_mae,
            "boundary_anchor_count": len(anchors),
            "boundary_positive_rows": int(boundary_labels.sum()),
            "boundary_negative_rows": int((~boundary_labels).sum()),
            "boundary_auc": auc,
            "accepted": task_accept,
        }

    validation_rows = np.concatenate(all_validation_rows)
    stream_report = validate_streaming_events(labels=labels, rows=validation_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readout_path = args.output_dir / "readout.npz"
    temporary = args.output_dir / f".readout.{os.getpid()}.tmp.npz"
    np.savez_compressed(temporary, **packed)
    temporary.replace(readout_path)
    report = {
        "schema_version": 1,
        "protocol": "pi05_r2_causal_recurrence_readout_v1",
        "causal": True,
        "future_observation_used": False,
        "encoder": "frozen dinov3-base pooled patch grid",
        "tasks": reports,
        "streaming_event_validation": stream_report,
        "acceptance": {
            "max_progress_mae": MAX_FIELD_MAE,
            "max_density_mae": MAX_FIELD_MAE,
            "minimum_boundary_auc": MIN_BOUNDARY_AUC,
            "minimum_boundary_class_rows": 5,
            "all_tasks_must_pass": True,
            "accepted": accepted,
        },
        "schedule": R2ScheduleConfig().__dict__,
        "source_sha256": {
            "selection": sha256(args.selection),
            "labels_manifest": sha256(args.labels_manifest),
            "labels": sha256(args.labels),
            "probe_labels": sha256(args.probe_labels),
            "reference_trajectories": sha256(args.reference_trajectories),
        },
        "readout_sha256": sha256(readout_path),
    }
    atomic_text(
        args.output_dir / "readout_manifest.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    gate = args.output_dir / ("r2_readout.accepted" if accepted else "r2_readout.rejected")
    for stale in args.output_dir.glob("r2_readout.*"):
        stale.unlink()
    atomic_text(gate, json.dumps(report["acceptance"], sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

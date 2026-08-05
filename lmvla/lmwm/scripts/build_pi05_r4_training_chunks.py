#!/usr/bin/env python3
"""Build audited query-level pi0.5 action chunks for the R4 matched arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audit_pi05_r4_query_dataset import audit


CHUNK_SIZE = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def outcome_weights(tasks: np.ndarray, success: np.ndarray, temperature: float) -> np.ndarray:
    """Exponentiate centered terminal return and normalize within each task.

    These are episode-outcome weights, not estimates of action advantage. The
    normalization preserves the mean loss scale of every task.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("outcome temperature must be finite and positive")
    weights = np.empty(len(success), dtype=np.float32)
    for task in np.unique(tasks):
        mask = tasks == task
        values = success[mask].astype(np.float64)
        raw = np.exp((values - values.mean()) / temperature)
        weights[mask] = (raw / raw.mean()).astype(np.float32)
    return weights


def build(
    query_manifest: Path,
    outcome_manifest: Path,
    output: Path,
    *,
    outcome_temperature: float = 1.0,
) -> dict:
    gate = audit(query_manifest, outcome_manifest)
    if not gate["accepted"]:
        raise ValueError("R4 query/outcome audit is not accepted")

    query = json.loads(query_manifest.read_text(encoding="utf-8"))
    records = sorted(
        query["records"], key=lambda row: (str(row["task"]), int(row["scene_seed"]))
    )
    task_names = sorted({str(row["task"]) for row in records})
    task_to_id = {task: index for index, task in enumerate(task_names)}
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    tasks: list[str] = []
    task_ids: list[int] = []
    scene_seeds: list[int] = []
    successes: list[bool] = []
    record_indices: list[int] = []
    query_indices: list[int] = []
    query_frames_out: list[int] = []
    query_artifacts: list[str] = []
    instructions: list[str] = []

    for record_index, record in enumerate(records):
        trajectory_path = resolve_artifact(query_manifest, str(record["trajectory"]))
        query_path = resolve_artifact(query_manifest, str(record["query_observations"]))
        with np.load(trajectory_path, allow_pickle=False) as payload:
            trajectory_actions = np.asarray(payload["actions"], dtype=np.float32)
        with np.load(query_path, allow_pickle=False) as payload:
            query_frames = np.asarray(payload["query_frame_index"], dtype=np.int64)
            query_states = np.asarray(payload["query_states"], dtype=np.float32)
            instruction = str(np.asarray(payload["instruction"]).item()).strip()
        for query_index, (frame, state) in enumerate(
            zip(query_frames, query_states, strict=True)
        ):
            frame = int(frame)
            valid = min(CHUNK_SIZE, len(trajectory_actions) - frame)
            if valid <= 0:
                raise ValueError(f"empty action chunk for record {record_index} frame {frame}")
            chunk = np.empty((CHUNK_SIZE, 14), dtype=np.float32)
            chunk[:valid] = trajectory_actions[frame : frame + valid]
            chunk[valid:] = chunk[valid - 1]
            mask = np.zeros(CHUNK_SIZE, dtype=bool)
            mask[:valid] = True
            states.append(state)
            actions.append(chunk)
            action_masks.append(mask)
            task = str(record["task"])
            tasks.append(task)
            task_ids.append(task_to_id[task])
            scene_seeds.append(int(record["scene_seed"]))
            successes.append(bool(record["success"]))
            record_indices.append(record_index)
            query_indices.append(query_index)
            query_frames_out.append(frame)
            query_artifacts.append(os.path.relpath(query_path, output.parent))
            instructions.append(instruction)

    task_array = np.asarray(tasks)
    success_array = np.asarray(successes, dtype=bool)
    calibrated = outcome_weights(task_array, success_array, outcome_temperature)
    arrays = {
        "state": np.stack(states),
        "action": np.stack(actions),
        "action_valid": np.stack(action_masks),
        "task": task_array,
        "task_id": np.asarray(task_ids, dtype=np.int16),
        "scene_seed": np.asarray(scene_seeds, dtype=np.int64),
        "success": success_array,
        "record_index": np.asarray(record_indices, dtype=np.int32),
        "query_index": np.asarray(query_indices, dtype=np.int16),
        "query_frame": np.asarray(query_frames_out, dtype=np.int32),
        "query_observations": np.asarray(query_artifacts),
        "instruction": np.asarray(instructions),
        "ordinary_weight": np.ones(len(states), dtype=np.float32),
        "outcome_calibrated_weight": calibrated,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)

    per_task = {}
    for task in task_names:
        mask = task_array == task
        per_task[task] = {
            "episodes": sum(str(row["task"]) == task for row in records),
            "queries": int(mask.sum()),
            "success_queries": int(success_array[mask].sum()),
            "failure_queries": int((~success_array[mask]).sum()),
            "outcome_weight_mean": float(calibrated[mask].mean()),
            "outcome_weight_min": float(calibrated[mask].min()),
            "outcome_weight_max": float(calibrated[mask].max()),
        }
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_query_action_chunks_v1",
        "interpretation": (
            "outcome_calibrated_weight is centered terminal-return weighting; "
            "it is not an action-advantage, Q-value, or world-critic estimate"
        ),
        "query_manifest": str(query_manifest.resolve()),
        "query_manifest_sha256": sha256(query_manifest),
        "outcome_manifest": str(outcome_manifest.resolve()),
        "outcome_manifest_sha256": sha256(outcome_manifest),
        "behavior_policy_sha256": str(query["behavior_policy_sha256"]),
        "record_count": len(records),
        "sample_count": len(states),
        "chunk_size": CHUNK_SIZE,
        "outcome_temperature": outcome_temperature,
        "task_to_id": task_to_id,
        "per_task": per_task,
        "chunks": str(output.resolve()),
        "chunks_sha256": sha256(output),
        "source_audit": gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--outcome-temperature", type=float, default=1.0)
    args = parser.parse_args()
    report = build(
        args.query_manifest.resolve(),
        args.outcome_manifest.resolve(),
        args.output.resolve(),
        outcome_temperature=args.outcome_temperature,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"accepted": True, "samples": report["sample_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

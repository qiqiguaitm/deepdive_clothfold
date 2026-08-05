#!/usr/bin/env python3
"""Audit trainable three-camera policy-query records for R4."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REQUIRED_TASKS = {
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(record: dict) -> tuple[str, int, bool, str]:
    return (
        str(record["task"]),
        int(record["scene_seed"]),
        bool(record["success"]),
        str(record["behavior_policy_sha256"]),
    )


def audit(query_manifest: Path, outcome_manifest: Path) -> dict:
    query = json.loads(query_manifest.read_text(encoding="utf-8"))
    outcome = json.loads(outcome_manifest.read_text(encoding="utf-8"))
    expected = {
        identity(record)
        for record in outcome.get("records", [])
        if str(record.get("split")) == "train"
    }
    records = query.get("records", [])
    observed = set()
    errors = []
    support = defaultdict(lambda: {"success": 0, "failure": 0})
    query_count = 0
    action_count = 0
    ignored_unexecuted_query_count = 0
    root = query_manifest.parent
    for index, record in enumerate(records):
        try:
            if str(record["split"]) != "train":
                raise ValueError("query-training manifest contains a non-train record")
            task = str(record["task"])
            if task not in REQUIRED_TASKS:
                raise ValueError(f"unexpected task {task}")
            record_identity = identity(record)
            if record_identity in observed:
                raise ValueError(f"duplicate record identity {record_identity}")
            observed.add(record_identity)
            trajectory = root / str(record["trajectory"])
            queries = root / str(record["query_observations"])
            if not trajectory.is_file() or not queries.is_file():
                raise FileNotFoundError("missing trajectory or query observations")
            if sha256(trajectory) != str(record["trajectory_sha256"]):
                raise ValueError("trajectory hash mismatch")
            if sha256(queries) != str(record["query_observations_sha256"]):
                raise ValueError("query observation hash mismatch")
            with np.load(trajectory, allow_pickle=False) as payload:
                actions = np.asarray(payload["actions"])
                states = np.asarray(payload["states"])
                frames = np.asarray(payload["frame_index"], dtype=np.int64)
            with np.load(queries, allow_pickle=False) as payload:
                query_frames = np.asarray(payload["query_frame_index"], dtype=np.int64)
                query_states = np.asarray(payload["query_states"])
                images = {name: np.asarray(payload[name]) for name in CAMERAS}
                instruction = str(np.asarray(payload["instruction"]).item()).strip()
            if actions.shape != states.shape or actions.ndim != 2 or actions.shape[1] != 14:
                raise ValueError("actions/states must have matching [T,14] shape")
            if not np.array_equal(frames, np.arange(len(frames))):
                raise ValueError("trajectory frame_index must be contiguous from zero")
            if query_frames.ndim != 1 or len(query_frames) == 0 or query_frames[0] != 0:
                raise ValueError("query frames must be a nonempty sequence beginning at zero")
            if np.any(np.diff(query_frames) != 50):
                raise ValueError("policy query intervals must equal the frozen 50-step replan horizon")
            if query_frames[-1] >= len(actions):
                invalid = query_frames >= len(actions)
                if invalid.sum() != 1 or query_frames[-1] != len(actions):
                    raise ValueError("query frame lies outside the executed trajectory")
                query_frames = query_frames[:-1]
                query_states = query_states[:-1]
                images = {name: values[:-1] for name, values in images.items()}
                ignored_unexecuted_query_count += 1
                if len(query_frames) == 0:
                    raise ValueError("trajectory contains no executed policy query")
            if query_states.shape != (len(query_frames), 14):
                raise ValueError("query states must have [Q,14] shape")
            if not np.allclose(query_states, states[query_frames], rtol=0.0, atol=1e-6):
                raise ValueError("query states do not align with per-step trajectory states")
            for name, values in images.items():
                if (
                    values.dtype != np.uint8
                    or values.ndim != 4
                    or values.shape[0] != len(query_frames)
                    or values.shape[-1] != 3
                ):
                    raise ValueError(f"invalid query camera array {name}: {values.shape}/{values.dtype}")
            if len({values.shape[1:] for values in images.values()}) != 1:
                raise ValueError("three query cameras do not share a common image shape")
            if not instruction or instruction == "None":
                raise ValueError("query instruction is missing")
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError("trajectory contains nonfinite values")
            support[task]["success" if bool(record["success"]) else "failure"] += 1
            query_count += len(query_frames)
            action_count += len(actions)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")

    checks = {
        "all_records_valid": not errors,
        "exact_predeclared_train_scene_outcomes": observed == expected,
        "all_six_tasks_present": set(support) == REQUIRED_TASKS,
        "success_and_failure_support_per_task": all(
            support[task][label] >= 1
            for task in REQUIRED_TASKS
            for label in ("success", "failure")
        ),
        "three_camera_policy_queries_aligned": query_count > 0 and not errors,
        "behavior_policy_matches_outcome_manifest": str(query.get("behavior_policy_sha256"))
        == str(outcome.get("behavior_policy_sha256")),
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_policy_query_dataset_audit_v1",
        "query_manifest": str(query_manifest.resolve()),
        "outcome_manifest": str(outcome_manifest.resolve()),
        "record_count": len(records),
        "expected_train_record_count": len(expected),
        "query_count": query_count,
        "executed_action_count": action_count,
        "ignored_unexecuted_query_count": ignored_unexecuted_query_count,
        "support": dict(support),
        "errors": errors,
        "checks": checks,
        "accepted": bool(all(checks.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--outcome-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.query_manifest.resolve(), args.outcome_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

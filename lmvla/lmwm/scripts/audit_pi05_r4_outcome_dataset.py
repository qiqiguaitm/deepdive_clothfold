#!/usr/bin/env python3
"""Audit the action-bearing outcome dataset required before pi0.5 R4 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_TASKS = {
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
}
HEX_DIGITS = set("0123456789abcdef")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value.lower()) <= HEX_DIGITS


def audit(
    manifest_path: Path,
    *,
    minimum_outcomes_per_task: int = 1,
    require_eval_split: bool = True,
    expected_record_count: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("R4 manifest must contain nonempty records")
    if minimum_outcomes_per_task < 1:
        raise ValueError("minimum outcomes per task must be positive")

    root = manifest_path.parent
    identities: set[tuple[str, str, int]] = set()
    split_scenes: dict[tuple[str, int], str] = {}
    support: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"success": 0, "failure": 0})
    )
    transition_count = 0
    behavior_policies: set[str] = set()
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            task = str(record["task"])
            split = str(record["split"])
            episode_id = int(record["episode_id"])
            scene_seed = int(record["scene_seed"])
            success = bool(record["success"])
            behavior_policy = str(record["behavior_policy_sha256"])
            if task not in REQUIRED_TASKS:
                raise ValueError(f"unexpected task {task}")
            if split not in {"train", "eval"}:
                raise ValueError(f"invalid split {split}")
            if not valid_digest(behavior_policy):
                raise ValueError("invalid behavior-policy digest")
            identity = (task, split, episode_id)
            if identity in identities:
                raise ValueError(f"duplicate identity {identity}")
            identities.add(identity)
            scene_identity = (task, scene_seed)
            previous_split = split_scenes.setdefault(scene_identity, split)
            if previous_split != split:
                raise ValueError(f"scene leakage across splits: {scene_identity}")

            trajectory = root / record["trajectory"]
            video = root / record["video"]
            if not trajectory.is_file() or not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError("missing trajectory or video")
            if sha256(trajectory) != record["trajectory_sha256"]:
                raise ValueError("trajectory hash mismatch")
            if sha256(video) != record["video_sha256"]:
                raise ValueError("video hash mismatch")
            with np.load(trajectory, allow_pickle=False) as payload:
                arrays = {name: np.asarray(payload[name]) for name in ("actions", "states", "frame_index")}
            lengths = {array.shape[0] for array in arrays.values() if array.ndim >= 1}
            if any(array.ndim < 1 for array in arrays.values()) or len(lengths) != 1:
                raise ValueError("actions, states, and frame_index must be aligned")
            length = lengths.pop()
            if length < 2 or not all(np.isfinite(array).all() for array in arrays.values()):
                raise ValueError("trajectory arrays must be finite and contain at least two steps")
            if np.any(np.diff(arrays["frame_index"].astype(np.int64)) <= 0):
                raise ValueError("frame_index must be strictly increasing")
            transition_count += int(length)
            behavior_policies.add(behavior_policy)
            support[split][task]["success" if success else "failure"] += 1
        except (KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")

    checks = {
        "all_records_valid": not errors,
        "expected_record_count": (
            expected_record_count is None or len(records) == expected_record_count
        ),
        "all_six_tasks_present_in_train": set(support["train"]) == REQUIRED_TASKS,
        "train_has_success_and_failure_support_per_task": all(
            support["train"][task][outcome] >= minimum_outcomes_per_task
            for task in REQUIRED_TASKS
            for outcome in ("success", "failure")
        ),
        "eval_split_present": (
            not require_eval_split or set(support["eval"]) == REQUIRED_TASKS
        ),
        "train_eval_scene_disjoint": not any("scene leakage" in error for error in errors),
        "behavior_policy_identity_present": bool(behavior_policies),
        "action_state_observation_alignment_present": transition_count > 0 and not errors,
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_action_bearing_outcome_dataset_audit_v1",
        "manifest": str(manifest_path.resolve()),
        "minimum_successes_and_failures_per_train_task": minimum_outcomes_per_task,
        "require_eval_split": require_eval_split,
        "expected_record_count": expected_record_count,
        "record_count": len(records),
        "transition_count": transition_count,
        "behavior_policy_sha256": sorted(behavior_policies),
        "support": {split: dict(tasks) for split, tasks in support.items()},
        "errors": errors,
        "checks": checks,
        "accepted": bool(all(checks.values())),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-outcomes-per-task", type=int, default=1)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--expected-record-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.manifest.resolve(),
        minimum_outcomes_per_task=args.minimum_outcomes_per_task,
        require_eval_split=not args.train_only,
        expected_record_count=args.expected_record_count,
    )
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

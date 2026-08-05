#!/usr/bin/env python3
"""Validate frozen MT3 protocol paths, hashes, and non-tunable invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_invariants(protocol: dict, split: dict, audit: dict) -> None:
    if len(split["train_episodes"]) != 960 or len(split["val_episodes"]) != 240:
        raise ValueError("tracker split size differs from the frozen 960/240 protocol")
    if set(split["train_episodes"]).intersection(split["val_episodes"]):
        raise ValueError("tracker split contains episode leakage")
    if protocol["shared_encoder"]["external_encoder"] is not None:
        raise ValueError("MT3 protocol must not introduce an external visual encoder")
    if "raw pi05_base" not in protocol["shared_encoder"]["tracker_only_checkpoint"]:
        raise ValueError("tracker-only features must use the raw pi05_base joint-policy initialization")
    if protocol["tracker_only_training"]["selection_uses_closed_loop_success"]:
        raise ValueError("tracker selection must not use closed-loop success")
    joint = protocol["joint_policy_training"]
    if joint["vision_encoder_frozen"] or not joint["tracker_receives_live_current_encoder_features"]:
        raise ValueError("joint MT3 training must track the live unfrozen pi0.5 encoder")
    if joint.get("tracker_gradient_into_vision_encoder") is not False:
        raise ValueError("MT3 tracker gradients must be isolated from the vision encoder")
    if joint["policy_updates"] != 50000 or joint["policy_batch_size"] != 16:
        raise ValueError("joint policy recipe differs from the frozen A0 update/batch protocol")
    routing = joint["routing_scope"]
    ranges = sorted(
        (*bounds, int(task)) for task, bounds in routing["episode_ranges_by_task_id"].items()
    )
    if set(routing["episode_ranges_by_task_id"]) != {str(value) for value in range(6)}:
        raise ValueError("joint routing scope must cover all six transition task IDs")
    if any(left[1] >= right[0] for left, right in zip(ranges, ranges[1:], strict=False)):
        raise ValueError("joint routing episode intervals overlap")
    if sum(upper - lower + 1 for lower, upper, _ in ranges) != routing["covered_episodes"]:
        raise ValueError("joint routing episode count is inconsistent")
    if routing["outside_scope_transition"] != "null":
        raise ValueError("tasks outside the six-task panel must use the null transition")
    if protocol["tracker_only_training"]["random_seed"] != 1000:
        raise ValueError("tracker-only selection seed differs from the frozen protocol")
    required_checks = {
        "episode_frame_unique",
        "frame_within_episode_length",
        "parquet_complete",
        "three_view_video_complete",
        "base_history_cache_complete",
    }
    if not required_checks.issubset(audit["checks"]):
        raise ValueError("MT3 data audit is missing required completeness checks")
    if not all(audit["checks"][name] for name in required_checks):
        raise ValueError("MT3 data audit reports incomplete tracker inputs")
    if audit["checks"]["episode_split_leakage"]:
        raise ValueError("MT3 data audit reports episode split leakage")
    if audit["episodes"] != {"total": 1200, "train": 960, "validation": 240}:
        raise ValueError("MT3 data audit episode counts differ from the frozen split")
    if audit["rows"]["total"] != audit["rows"]["train"] + audit["rows"]["validation"]:
        raise ValueError("MT3 data audit row counts are inconsistent")
    expected_history = audit["rows"]["total"] * len(audit["history_offsets_at_50hz"])
    if audit["rows"]["history_frame_references"] != expected_history:
        raise ValueError("MT3 history-frame audit count is inconsistent")


def validate(protocol: dict, repo: Path) -> None:
    sources = protocol["sources"]
    checks = (
        ("base_checkpoint", "base_checkpoint_metadata_sha256", "_METADATA"),
        ("normalization", "normalization_sha256", None),
        ("transition_pairs", "transition_pairs_sha256", None),
        ("tracker_split", "tracker_split_sha256", None),
        ("data_audit", "data_audit_sha256", None),
    )
    for path_key, hash_key, suffix in checks:
        path = repo / sources[path_key]
        if suffix is not None:
            path /= suffix
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != sources[hash_key]:
            raise ValueError(f"hash mismatch for {path_key}: {path}")

    split = json.loads((repo / sources["tracker_split"]).read_text())
    audit = json.loads((repo / sources["data_audit"]).read_text())
    if audit["split_sha256"] != sources["tracker_split_sha256"]:
        raise ValueError("data audit and protocol reference different tracker splits")
    if audit["pairs_sha256"] != sources["transition_pairs_sha256"]:
        raise ValueError("data audit and protocol reference different transition pairs")
    validate_invariants(protocol, split, audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    validate(json.loads(args.protocol.read_text()), args.repo.resolve())
    print(f"validated={args.protocol}")


if __name__ == "__main__":
    main()

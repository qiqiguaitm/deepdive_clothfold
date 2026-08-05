#!/usr/bin/env python3
"""Freeze the recurrence-aligned pi0.5 R1 four-arm protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dense-targets", type=Path, required=True)
    parser.add_argument("--dense-targets-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scene = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    if len(scene.get("eval_seeds", {})) != 4:
        raise ValueError("R1 requires four frozen evaluation seeds")
    task_maps = list(scene["eval_seeds"].values())
    if not task_maps or any(len(task_map) != 6 for task_map in task_maps):
        raise ValueError("R1 requires the frozen six-task scene manifest")
    source_hashes = {str(path.resolve().relative_to(repo)): sha256(path) for path in args.source}
    dense_manifest = json.loads(args.dense_targets_manifest.read_text(encoding="utf-8"))
    if dense_manifest.get("dense_targets_sha256") != sha256(args.dense_targets):
        raise ValueError("R1 dense-target manifest does not match the target artifact")
    if dense_manifest.get("horizon_frames") != 50:
        raise ValueError("R1 dense targets require the frozen 50-frame horizon")
    if dense_manifest.get("episode_count") != 1200:
        raise ValueError("R1 dense targets require all 1,200 reference episodes")
    if dense_manifest.get("physical_task_count") != 6:
        raise ValueError("R1 dense targets require all six physical tasks")
    if int(dense_manifest.get("target_rows", 0)) < 300_000:
        raise ValueError("R1 dense targets are unexpectedly sparse")
    payload = {
        "schema_version": 1,
        "protocol": "pi05_r1_recurrence_aligned_four_arm_v1",
        "seed1000_arms": ["a0", "predictive", "crave", "combined"],
        "causal_interventions": ["combined_zero_gate", "combined_shuffled"],
        "conditional_replication": {
            "required_gate": "seed1000 accepted",
            "training_seeds": [1001, 1002],
            "arms_per_seed": ["a0", "predictive", "crave", "combined"],
            "analysis": "hierarchical paired bootstrap over training seed and episode",
        },
        "checkpoint_reuse": {
            "a0": "matched P1 current-source A0",
            "predictive": "matched P1 normal predictive adapter",
        },
        "teacher": {
            "source": "CRAVE-v2 R0 dense reference-trajectory targets",
            "artifact": str(args.dense_targets.resolve().relative_to(repo)),
            "artifact_manifest": str(
                args.dense_targets_manifest.resolve().relative_to(repo)
            ),
            "targets": [
                "fixed-horizon progress-change distribution",
                "future recurrence-density distribution",
                "phase-boundary crossing",
            ],
            "visual_gradient": "stop before native pi0.5 visual tokens",
        },
        "route": "zero-initialized per-action-token route into action expert only",
        "scene_manifest": str(args.scene_manifest.resolve().relative_to(repo)),
        "scene_manifest_sha256": sha256(args.scene_manifest),
        "acceptance_gate": {
            "bootstrap_resamples": 20000,
            "bootstrap_seed": 20260804,
            "required_ci_lower_positive_vs": [
                "a0",
                "predictive",
                "crave",
                "zero_route",
                "shuffled_action",
            ],
            "max_task_regression_vs_a0": -0.05,
        },
        "immutable_artifact_sha256": {
            "base_params_metadata": sha256(args.base_metadata),
            "norm_stats": sha256(args.norm_stats),
            "dense_targets": sha256(args.dense_targets),
            "dense_targets_manifest": sha256(args.dense_targets_manifest),
        },
        "source_sha256": source_hashes,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

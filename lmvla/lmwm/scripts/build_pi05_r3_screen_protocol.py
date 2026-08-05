#!/usr/bin/env python3
"""Freeze the five-arm, same-scene R3 privileged-semantic screening protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CONDITIONS = {
    "semantic_next": {"prompt_mode": "semantic-next", "tracker_intervention": "correct"},
    "generic_stage": {"prompt_mode": "generic-stage", "tracker_intervention": "correct"},
    "semantic_current": {"prompt_mode": "semantic-current", "tracker_intervention": "correct"},
    "shuffled_semantic": {"prompt_mode": "semantic-next", "tracker_intervention": "within-task"},
    "no_subtask": {"prompt_mode": "none", "tracker_intervention": "correct"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_manifest(source: dict, episodes_per_cell: int) -> dict:
    eval_seeds = source.get("eval_seeds")
    if not isinstance(eval_seeds, dict) or not eval_seeds:
        raise ValueError("source manifest has no eval_seeds")
    frozen = {}
    for eval_seed, task_map in sorted(eval_seeds.items(), key=lambda item: int(item[0])):
        if not isinstance(task_map, dict):
            raise ValueError(f"eval seed {eval_seed} is not a task map")
        frozen[str(eval_seed)] = {}
        for task, seeds in sorted(task_map.items()):
            if len(seeds) < episodes_per_cell:
                raise ValueError(f"{eval_seed}/{task} has only {len(seeds)} scene seeds")
            frozen[str(eval_seed)][task] = [int(value) for value in seeds[:episodes_per_cell]]
    return {
        "schema_version": 1,
        "protocol": "pi05_r3_semantic_screen_same_scene_v1",
        "episodes_per_cell": episodes_per_cell,
        "eval_seeds": frozen,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--scene-manifest-out", type=Path, required=True)
    parser.add_argument("--protocol-out", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=10)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--checkpoint-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes_per_cell <= 0:
        raise ValueError("episodes-per-cell must be positive")

    source_payload = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest = build_manifest(source_payload, args.episodes_per_cell)
    atomic_json(args.scene_manifest_out, manifest)
    protocol = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "checkpoint": "SidneyXie/pi05_robotwin public A0, unchanged for all arms",
        "conditions": CONDITIONS,
        "tasks": sorted(next(iter(manifest["eval_seeds"].values()))),
        "eval_seeds": sorted(int(value) for value in manifest["eval_seeds"]),
        "episodes_per_cell": args.episodes_per_cell,
        "scene_cells": sum(len(tasks) for tasks in manifest["eval_seeds"].values()),
        "scene_manifest_sha256": sha256(args.scene_manifest_out),
        "source_scene_manifest": str(args.source_manifest.resolve()),
        "source_scene_manifest_sha256": sha256(args.source_manifest),
        "prompt_template": {
            "semantic_next": "<official instruction> Next subtask: <source-grounded event>.",
            "semantic_current": "<official instruction> Current subtask: <source-grounded event>.",
            "generic_stage": "<official instruction> Current stage ID: <i>. Next stage ID: <j>.",
            "no_subtask": "<official instruction>",
        },
        "oracle_scope": "same-scene expert joint trace supplies monotonic phase only; no future policy rollout state",
        "matched_controls": [
            "identical public A0 weights and tokenizer",
            "identical fixed scene seeds",
            "identical replan horizon and action budget",
            "identical oracle tracker execution in all five arms",
        ],
        "acceptance_gate": {
            "primary": "paired episode-bootstrap 95% lower bound for semantic_next - no_subtask > 0",
            "controls": "semantic_next point estimate exceeds generic_stage, semantic_current, and shuffled_semantic",
            "task_safety": "no per-task semantic_next - no_subtask regression below -0.05",
            "bootstrap_resamples": 20000,
            "bootstrap_seed": 20260804,
        },
        "source_sha256": {str(path): sha256(path) for path in args.source},
        "public_artifact_sha256": {
            "config.json": sha256(args.checkpoint_config),
            "model.safetensors": sha256(args.checkpoint_weights),
            "tokenizer.model": sha256(args.tokenizer_model),
        },
    }
    atomic_json(args.protocol_out, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

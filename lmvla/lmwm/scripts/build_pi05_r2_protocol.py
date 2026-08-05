#!/usr/bin/env python3
"""Freeze the public-pi0.5 same-scene R2 adaptive-execution protocol."""

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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--checkpoint-weights", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scene = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    eval_seeds = scene["eval_seeds"]
    tasks = sorted(next(iter(eval_seeds.values())))
    if len(eval_seeds) != 4 or len(tasks) != 6:
        raise ValueError("R2 requires the frozen 24-cell six-task scene manifest")
    if any(len(values) != 10 for task_map in eval_seeds.values() for values in task_map.values()):
        raise ValueError("R2 requires exactly ten episodes per cell")
    source_hashes = {}
    for path in args.source:
        relative = str(path.resolve().relative_to(repo))
        source_hashes[relative] = sha256(path)
    payload = {
        "schema_version": 1,
        "protocol": "pi05_r2_adaptive_execution_same_scene_v1",
        "checkpoint": "SidneyXie/pi05_robotwin public A0, unchanged in both arms",
        "conditions": {
            "fixed4": {"execution_horizon": 4, "online_dino": False},
            "adaptive": {
                "execution_horizons": [1, 2, 4, 8],
                "online_dino": True,
                "future_observation_used": False,
            },
        },
        "tasks": tasks,
        "eval_seeds": sorted(map(int, eval_seeds)),
        "episodes_per_cell": 10,
        "scene_cells": 24,
        "scene_manifest": str(args.scene_manifest.resolve().relative_to(repo)),
        "scene_manifest_sha256": sha256(args.scene_manifest),
        "action_horizon": 8,
        "query_budget": "fixed-four allowance with at most one borrowed event query; stable eight-step chunks repay debt",
        "acceptance_gate": {
            "primary": "paired episode-bootstrap 95% lower bound for adaptive - fixed4 success > 0",
            "task_safety": "no per-task regression below -0.05",
            "compute": "adaptive aggregate policy-query count <= 1.05 times fixed4",
            "bootstrap_resamples": 20000,
            "bootstrap_seed": 20260804,
        },
        "readout_sources": {
            "selection": "lmvla/lmwm/data/pi05_crave_r0_v1/selection_manifest.json",
            "labels_manifest": "lmvla/lmwm/data/pi05_crave_r0_v1/labels_manifest.json",
            "labels": "lmvla/lmwm/data/pi05_crave_r0_v1/labels.npz",
            "probe_labels": "lmvla/lmwm/data/pi05_crave_r0_v1/probe_train.npz",
            "reference_trajectories": "lmvla/lmwm/data/pi05_crave_r0_v1/reference_trajectories.npz",
        },
        "source_sha256": source_hashes,
        "public_artifact_sha256": {
            "config.json": sha256(args.checkpoint_config),
            "model.safetensors": sha256(args.checkpoint_weights),
            "tokenizer.model": sha256(args.tokenizer_model),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize non-gating pi0.5 step-40k safety probes against final A0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = ("beat_block_hammer", "stack_blocks_three")
RESULTS = {
    "a0_final": "pi05_rt_a0_public_exact_seed1000",
    "a2_abs_40k": "pi05_a2_abs_seed1000_step40000_{task}_probe_safety40k",
    "a3_live_40k": "pi05_a3_live_seed1000_step40000_{task}_probe_safety40k",
}


class MissingCellError(ValueError):
    pass


def load_cell(root: Path, task: str, expected_seeds: list[int], manifest_sha: str) -> dict:
    paths = sorted(root.glob(f"seed0/**/tasks/{task}/summary.json"))
    if not paths:
        raise MissingCellError(f"missing {task} summary under {root}")
    if len(paths) != 1:
        raise ValueError(f"expected one {task} summary under {root}, found {len(paths)}")
    payload = json.loads(paths[0].read_text())
    episodes = payload.get("episodes", [])
    seeds = [int(episode["seed"]) for episode in episodes]
    if seeds != expected_seeds:
        raise ValueError(f"{paths[0]} does not match the frozen seed order")
    manifest = payload.get("fixed_seed_manifest") or {}
    if manifest.get("sha256") != manifest_sha:
        raise ValueError(f"{paths[0]} manifest SHA mismatch")
    outcomes = {int(episode["seed"]): bool(episode["success"]) for episode in episodes}
    if len(outcomes) != len(expected_seeds):
        raise ValueError(f"{paths[0]} has duplicate episode seeds")
    return {
        "successes": sum(outcomes.values()),
        "attempts": len(outcomes),
        "success_rate": sum(outcomes.values()) / len(outcomes),
        "outcomes": outcomes,
        "summary": str(paths[0]),
    }


def summarize(repo: Path) -> dict:
    manifest_path = repo / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    expected = {
        task: [int(seed) for seed in manifest["eval_seeds"]["0"][task]]
        for task in TASKS
    }
    result_base = repo / "lmvla/lawam/results/eval_runs/robotwin"
    cells: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for arm, template in RESULTS.items():
        cells[arm] = {}
        for task in TASKS:
            root = result_base / template.format(task=task)
            try:
                cells[arm][task] = load_cell(root, task, expected[task], manifest_sha)
            except MissingCellError:
                missing.append(f"{arm}/{task}")

    comparisons: dict[str, dict[str, dict]] = {}
    if not missing:
        for arm in ("a2_abs_40k", "a3_live_40k"):
            comparisons[arm] = {}
            for task in TASKS:
                baseline = cells["a0_final"][task]
                treatment = cells[arm][task]
                paired = [
                    int(treatment["outcomes"][seed]) - int(baseline["outcomes"][seed])
                    for seed in expected[task]
                ]
                comparisons[arm][task] = {
                    "success_rate_delta_vs_a0": sum(paired) / len(paired),
                    "method_only_successes": sum(value == 1 for value in paired),
                    "a0_only_successes": sum(value == -1 for value in paired),
                    "paired_scenes": len(paired),
                }

    for arm_cells in cells.values():
        for cell in arm_cells.values():
            cell.pop("outcomes", None)
    return {
        "protocol": "step-40000 safety diagnostic; fixed eval seed 0; 50 scenes per task",
        "admissibility": "non-gating diagnostic; excluded from the step-49999 confirmatory matrix",
        "manifest_sha256": manifest_sha,
        "complete": not missing,
        "missing_cells": missing,
        "cells": cells,
        "paired_vs_a0": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.repo)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()

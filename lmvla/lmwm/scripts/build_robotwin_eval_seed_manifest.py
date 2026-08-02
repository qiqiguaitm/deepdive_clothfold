#!/usr/bin/env python3
"""Freeze accepted RoboTwin scene seeds from a complete evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DEFAULT_TASKS = (
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "stack_blocks_two",
    "blocks_ranking_size",
    "handover_block",
    "stack_blocks_three",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=50)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = parser.parse_args()

    cells: dict[tuple[int, str], tuple[list[int], Path]] = {}
    for summary_path in sorted(args.root.rglob("summary.json")):
        match = re.search(r"/seed(\d+)(?:/|$)", str(summary_path))
        if not match:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        task_name = payload.get("task_name")
        episodes = payload.get("episodes")
        if task_name not in args.tasks or not isinstance(episodes, list):
            continue
        eval_seed = int(match.group(1))
        if eval_seed not in args.eval_seeds:
            continue
        scene_seeds = [int(item["seed"]) for item in episodes]
        if len(scene_seeds) != args.episodes_per_cell:
            raise SystemExit(
                f"{summary_path}: expected {args.episodes_per_cell} episodes, got {len(scene_seeds)}"
            )
        if len(scene_seeds) != len(set(scene_seeds)):
            raise SystemExit(f"{summary_path}: duplicate scene seeds")
        key = (eval_seed, str(task_name))
        if key in cells:
            raise SystemExit(f"duplicate summary cell {key}: {cells[key][1]} and {summary_path}")
        cells[key] = (scene_seeds, summary_path)

    expected = {(seed, task) for seed in args.eval_seeds for task in args.tasks}
    missing = sorted(expected - set(cells))
    extra = sorted(set(cells) - expected)
    if missing or extra:
        raise SystemExit(f"manifest cell mismatch: missing={missing}, extra={extra}")

    manifest = {
        "version": 1,
        "source_root": str(args.root.resolve()),
        "episodes_per_cell": args.episodes_per_cell,
        "tasks": list(args.tasks),
        "eval_seeds": {
            str(eval_seed): {
                task: cells[(eval_seed, task)][0]
                for task in args.tasks
            }
            for eval_seed in args.eval_seeds
        },
        "source_summaries": {
            f"seed{eval_seed}:{task}": {
                "path": str(cells[(eval_seed, task)][1].resolve()),
                "sha256": hashlib.sha256(cells[(eval_seed, task)][1].read_bytes()).hexdigest(),
            }
            for eval_seed in args.eval_seeds
            for task in args.tasks
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells)}, sort_keys=True))


if __name__ == "__main__":
    main()

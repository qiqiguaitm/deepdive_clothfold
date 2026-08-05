#!/usr/bin/env python3
"""Freeze a small scene-disjoint subset for R4 trajectory collection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EVAL_SEEDS = (0, 1, 2, 3)


def build(source: dict, episodes_per_cell: int) -> dict:
    if episodes_per_cell < 1:
        raise ValueError("episodes_per_cell must be positive")
    source_seeds = source["eval_seeds"]
    result = {}
    seen: set[tuple[str, int]] = set()
    for eval_seed in EVAL_SEEDS:
        task_rows = source_seeds[str(eval_seed)]
        result[str(eval_seed)] = {}
        for task, values in sorted(task_rows.items()):
            selected = [int(value) for value in values[:episodes_per_cell]]
            if len(selected) != episodes_per_cell:
                raise ValueError(f"insufficient scenes for eval_seed={eval_seed} task={task}")
            for scene_seed in selected:
                identity = (str(task), scene_seed)
                if identity in seen:
                    raise ValueError(f"scene reused across eval seeds: {identity}")
                seen.add(identity)
            result[str(eval_seed)][str(task)] = selected
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_outcome_scene_seeds_v1",
        "episodes_per_cell": episodes_per_cell,
        "split_by_eval_seed": {"train": [0, 1], "eval": [2, 3]},
        "source_protocol": source.get("protocol"),
        "eval_seeds": result,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episodes-per-cell", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(json.loads(args.source.read_text()), args.episodes_per_cell)
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

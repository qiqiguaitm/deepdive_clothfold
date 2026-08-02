#!/usr/bin/env python3
"""Summarize a RoboTwin evaluation result tree into one reproducible JSON report."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def eval_seed(path: str) -> int | None:
    for pattern in (r"/seed(\d+)(?:/|$)", r"eval-seed(\d+)", r"-seed(\d+)(?:-|/)"):
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def summarize(root: Path) -> dict[str, Any]:
    paths = sorted(glob.glob(str(root / "**/summary.json"), recursive=True))
    records = []
    seen = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        task = data.get("task_name") or Path(path).parent.name
        seed = eval_seed(path)
        key = (task, seed)
        if key in seen:
            raise ValueError(f"duplicate task/eval-seed cell {key}: {path}")
        seen.add(key)
        episodes = int(data.get("n_episodes", len(data.get("episodes", []))))
        successes = int(
            data.get("successes", sum(bool(item.get("success")) for item in data.get("episodes", [])))
        )
        rate = float(data.get("success_rate", successes / episodes if episodes else 0.0))
        episode_outcomes = [
            {
                "episode_id": item.get("episode_id"),
                "scene_seed": item.get("seed"),
                "success": bool(item.get("success")),
            }
            for item in data.get("episodes", [])
        ]
        records.append(
            {
                "task": task,
                "eval_seed": seed,
                "successes": successes,
                "episodes": episodes,
                "success_rate": rate,
                "elapsed_sec": data.get("elapsed_sec"),
                "episode_outcomes": episode_outcomes,
                "path": path,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)
    tasks = {}
    for task, cells in sorted(grouped.items()):
        cells.sort(key=lambda item: (-1 if item["eval_seed"] is None else item["eval_seed"]))
        tasks[task] = {
            "mean_success_rate": mean(item["success_rate"] for item in cells),
            "total_successes": sum(item["successes"] for item in cells),
            "total_episodes": sum(item["episodes"] for item in cells),
            "cells": cells,
        }
    total_successes = sum(item["successes"] for item in records)
    total_episodes = sum(item["episodes"] for item in records)
    return {
        "root": str(root),
        "summary_count": len(records),
        "task_count": len(tasks),
        "macro_success_rate": mean(item["mean_success_rate"] for item in tasks.values()) if tasks else None,
        "micro_success_rate": total_successes / total_episodes if total_episodes else None,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-cells", type=int)
    args = parser.parse_args()
    report = summarize(args.root)
    if args.expected_cells is not None and report["summary_count"] < args.expected_cells:
        raise SystemExit(
            f"incomplete result tree: {report['summary_count']}/{args.expected_cells} summaries"
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

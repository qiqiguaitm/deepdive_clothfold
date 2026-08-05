#!/usr/bin/env python3
"""Summarize R2 outcomes while preserving per-cell policy-query and wall-clock costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from summarize_robotwin_eval import summarize


def eval_seed(path: Path) -> int:
    for part in path.parts:
        if re.fullmatch(r"seed\d+", part):
            return int(part.removeprefix("seed"))
    raise ValueError(f"cannot recover eval seed from {path}")


def summarize_r2(root: Path) -> dict:
    report = summarize(root)
    efficiency_cells = []
    for path in sorted(root.rglob("summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        episodes = int(payload["n_episodes"])
        efficiency_cells.append(
            {
                "task": str(payload["task_name"]),
                "eval_seed": eval_seed(path),
                "episodes": episodes,
                "model_queries": int(payload["model_queries"]),
                "queries_per_episode": float(payload["model_queries"]) / episodes,
                "elapsed_sec": float(payload["elapsed_sec"]),
                "elapsed_sec_per_episode": float(payload["elapsed_sec"]) / episodes,
                "total_action_steps": int(sum(int(row["steps"]) for row in payload["episodes"])),
                "path": str(path),
            }
        )
    if len(efficiency_cells) != report["summary_count"]:
        raise ValueError("R2 efficiency-cell coverage differs from outcome summary")
    report["efficiency_cells"] = efficiency_cells
    report["total_model_queries"] = sum(row["model_queries"] for row in efficiency_cells)
    report["total_elapsed_sec"] = sum(row["elapsed_sec"] for row in efficiency_cells)
    report["total_action_steps"] = sum(row["total_action_steps"] for row in efficiency_cells)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-cells", type=int, default=24)
    args = parser.parse_args()
    result = summarize_r2(args.root)
    if result["summary_count"] != args.expected_cells:
        raise ValueError(f"R2 cells {result['summary_count']} != {args.expected_cells}")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

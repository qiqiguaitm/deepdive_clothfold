#!/usr/bin/env python3
"""Apply the frozen MT3 pilot gate before learned-policy replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PREDECLARED_MULTISTAGE_TASKS = (
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
)
REQUIRED_CONTROLS = ("a0", "null", "within_task")


def decide(analysis: dict[str, Any]) -> dict[str, Any]:
    tests = {(row["control"], row["scope"]): row for row in analysis["tests"]}
    required = [
        (control, scope)
        for control in REQUIRED_CONTROLS
        for scope in ("pooled", *PREDECLARED_MULTISTAGE_TASKS)
    ]
    missing = [key for key in required if key not in tests]
    if missing:
        raise ValueError(f"analysis is missing MT3 pilot comparisons: {missing}")

    pooled = {
        control: float(tests[(control, "pooled")]["success_rate_delta"])
        for control in REQUIRED_CONTROLS
    }
    task_delta = {
        task: float(tests[("a0", task)]["success_rate_delta"])
        for task in PREDECLARED_MULTISTAGE_TASKS
    }
    improved = sorted(task for task, delta in task_delta.items() if delta > 0.0)
    checks = {
        "predicted_beats_a0": pooled["a0"] > 0.0,
        "predicted_beats_null": pooled["null"] > 0.0,
        "predicted_beats_within_task": pooled["within_task"] > 0.0,
        "at_least_two_multistage_tasks_beat_a0": len(improved) >= 2,
    }
    return {
        "accepted_for_replication": all(checks.values()),
        "scope": "MT3 seed-1000 pilot only; final claim requires the frozen three-seed interval",
        "checks": checks,
        "pooled_success_rate_delta": pooled,
        "predeclared_multistage_delta_vs_a0": task_delta,
        "improved_multistage_tasks": improved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    args = parser.parse_args()
    result = decide(json.loads(args.analysis.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.accepted_marker.unlink(missing_ok=True)
    if result["accepted_for_replication"]:
        args.accepted_marker.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_marker.write_text(f"accepted=true\ngate={args.output.resolve()}\n")


if __name__ == "__main__":
    main()

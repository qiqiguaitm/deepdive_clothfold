#!/usr/bin/env python3
"""Apply the frozen pilot gate before launching MT1 replication seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PREDECLARED_MULTISTAGE_TASKS = (
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
)
REQUIRED_POOLED_CONTROLS = ("a0", "null_input", "within_task", "null_trained")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def decide(analysis: dict[str, Any]) -> dict[str, Any]:
    tests = {(row["control"], row["scope"]): row for row in analysis["tests"]}
    missing = [
        (control, scope)
        for control in REQUIRED_POOLED_CONTROLS
        for scope in ("pooled", *PREDECLARED_MULTISTAGE_TASKS)
        if (control, scope) not in tests
    ]
    if missing:
        raise ValueError(f"analysis is missing gate comparisons: {missing}")

    pooled = {
        control: float(tests[(control, "pooled")]["success_rate_delta"])
        for control in REQUIRED_POOLED_CONTROLS
    }
    task_delta_vs_a0 = {
        task: float(tests[("a0", task)]["success_rate_delta"])
        for task in PREDECLARED_MULTISTAGE_TASKS
    }
    improved_tasks = sorted(task for task, delta in task_delta_vs_a0.items() if delta > 0.0)
    checks = {
        "correct_beats_a0": pooled["a0"] > 0.0,
        "correct_beats_null_input": pooled["null_input"] > 0.0,
        "correct_beats_within_task": pooled["within_task"] > 0.0,
        "correct_beats_null_trained": pooled["null_trained"] > 0.0,
        "at_least_two_multistage_tasks_beat_a0": len(improved_tasks) >= 2,
    }
    return {
        "accepted_for_replication": all(checks.values()),
        "scope": "pilot launch gate only; final claim requires the frozen three-seed hierarchical interval",
        "checks": checks,
        "pooled_success_rate_delta": pooled,
        "predeclared_multistage_delta_vs_a0": task_delta_vs_a0,
        "improved_multistage_tasks": improved_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    args = parser.parse_args()
    result = decide(json.loads(args.analysis.read_text()))
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.accepted_marker.unlink(missing_ok=True)
    if result["accepted_for_replication"]:
        atomic_write_text(
            args.accepted_marker,
            f"accepted=true\ngate={args.output.resolve()}\n",
        )


if __name__ == "__main__":
    main()

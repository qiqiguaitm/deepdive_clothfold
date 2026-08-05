#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


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


def outcome_map(report: dict[str, Any], task: str | None = None) -> dict[tuple[str, int, int], bool]:
    outcomes = {}
    for task_name, task_row in report["tasks"].items():
        if task is not None and task_name != task:
            continue
        for cell in task_row["cells"]:
            eval_seed = int(cell["eval_seed"])
            for episode in cell["episode_outcomes"]:
                key = (task_name, eval_seed, int(episode["scene_seed"]))
                if key in outcomes:
                    raise ValueError(f"duplicate outcome key: {key}")
                outcomes[key] = bool(episode["success"])
    return outcomes


def exact_mcnemar(candidate: dict, control: dict) -> dict[str, float | int]:
    if set(candidate) != set(control):
        raise ValueError("paired reports do not contain identical scene keys")
    keys = sorted(candidate)
    better = sum(candidate[key] and not control[key] for key in keys)
    worse = sum(control[key] and not candidate[key] for key in keys)
    discordant = better + worse
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(better, worse) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    candidate_rate = sum(int(candidate[key]) for key in keys) / len(keys)
    control_rate = sum(int(control[key]) for key in keys) / len(keys)
    return {
        "paired_episodes": len(keys),
        "candidate_success_rate": candidate_rate,
        "control_success_rate": control_rate,
        "candidate_only_success": better,
        "control_only_success": worse,
        "success_rate_delta": candidate_rate - control_rate,
        "mcnemar_exact_p": p_value,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["mcnemar_exact_p"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * rows[index]["mcnemar_exact_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def analyze(correct: dict[str, Any], controls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted(correct["tasks"])
    tests = []
    for control_name, control in controls.items():
        for task in [None, *tasks]:
            row = exact_mcnemar(outcome_map(correct, task), outcome_map(control, task))
            row.update({"control": control_name, "scope": "pooled" if task is None else task})
            tests.append(row)
    holm_adjust(tests)
    return {
        "correct_macro_success_rate": correct["macro_success_rate"],
        "controls": {name: report["macro_success_rate"] for name, report in controls.items()},
        "holm_family_size": len(tests),
        "holm_family_definition": (
            "all pooled and task-level correct-versus-control McNemar tests"
        ),
        "tests": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--correct", type=Path, required=True)
    parser.add_argument("--control", action="append", required=True, help="NAME=REPORT.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controls = {}
    for value in args.control:
        name, path = value.split("=", 1)
        controls[name] = json.loads(Path(path).read_text())
    result = analyze(json.loads(args.correct.read_text()), controls)
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

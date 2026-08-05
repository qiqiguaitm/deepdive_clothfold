#!/usr/bin/env python3
"""Apply the frozen seed-1000 recurrence-aligned four-arm R1 gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260804
MAX_TASK_REGRESSION = -0.05


def outcomes(report: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    if report.get("summary_count") != 24 or report.get("total_episodes") != 1200:
        raise ValueError("R1 reports require exactly 24 cells and 1,200 episodes")
    if len(report.get("tasks", {})) != 6:
        raise ValueError("R1 reports require six tasks")
    result = {}
    for task, task_payload in report["tasks"].items():
        for cell in task_payload["cells"]:
            eval_seed = int(cell["eval_seed"])
            for episode in cell["episode_outcomes"]:
                key = (str(task), eval_seed, int(episode["scene_seed"]))
                if key in result:
                    raise ValueError(f"duplicate R1 scene: {key}")
                result[key] = bool(episode["success"])
    if len(result) != 1200:
        raise ValueError(f"R1 report has {len(result)} unique scenes, expected 1,200")
    return result


def paired_interval(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1000):
        count = min(1000, BOOTSTRAP_RESAMPLES - start)
        index = rng.integers(0, len(delta), size=(count, len(delta)))
        samples[start : start + count] = delta[index].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def analyze(combined: dict, controls: dict[str, dict]) -> dict[str, Any]:
    combined_rows = outcomes(combined)
    control_rows = {name: outcomes(report) for name, report in controls.items()}
    if any(set(rows) != set(combined_rows) for rows in control_rows.values()):
        raise ValueError("R1 conditions do not contain identical scene identities")
    keys = sorted(combined_rows)
    candidate = np.asarray([combined_rows[key] for key in keys], dtype=np.float64)
    comparisons = {}
    for name, rows in control_rows.items():
        baseline = np.asarray([rows[key] for key in keys], dtype=np.float64)
        delta = candidate - baseline
        comparisons[name] = {
            "combined_success_rate": float(candidate.mean()),
            "control_success_rate": float(baseline.mean()),
            "delta": float(delta.mean()),
            "paired_bootstrap_ci95": paired_interval(delta),
        }
    task_deltas = {}
    a0 = control_rows["a0"]
    for task in sorted({key[0] for key in keys}):
        task_keys = [key for key in keys if key[0] == task]
        task_deltas[task] = float(
            np.mean([combined_rows[key] - a0[key] for key in task_keys])
        )
    required = ("a0", "predictive", "crave", "zero_route", "shuffled_action")
    checks = {
        f"combined_ci_lower_positive_vs_{name}": comparisons[name][
            "paired_bootstrap_ci95"
        ][0]
        > 0.0
        for name in required
    }
    checks["no_task_regression_below_minus_0_05_vs_a0"] = (
        min(task_deltas.values()) >= MAX_TASK_REGRESSION
    )
    accepted = bool(all(checks.values()))
    return {
        "schema_version": 1,
        "protocol": "pi05_r1_recurrence_aligned_seed1000_v1",
        "episodes": len(keys),
        "comparisons": comparisons,
        "task_deltas_vs_a0": task_deltas,
        "checks": checks,
        "accepted": accepted,
        "verdict": "accepted" if accepted else "rejected",
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--control", action="append", required=True, help="NAME=REPORT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    controls = {}
    for item in args.control:
        name, path = item.split("=", 1)
        controls[name] = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"a0", "predictive", "crave", "zero_route", "shuffled_action"}
    if set(controls) != required:
        raise ValueError(f"R1 controls must be {sorted(required)}, got {sorted(controls)}")
    result = analyze(json.loads(args.combined.read_text(encoding="utf-8")), controls)
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.gate_dir.glob("r1_gate.*"):
        stale.unlink()
    atomic_text(args.gate_dir / f"r1_gate.{result['verdict']}", f"report={args.output}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

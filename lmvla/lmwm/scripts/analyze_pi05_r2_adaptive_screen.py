#!/usr/bin/env python3
"""Analyze the preregistered same-scene R2 fixed4 versus adaptive screen."""

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
MAX_QUERY_RATIO = 1.05


def load_report(path: Path) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, float]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for task, task_payload in report["tasks"].items():
        for cell in task_payload["cells"]:
            eval_seed = int(cell["eval_seed"])
            for episode in cell["episode_outcomes"]:
                key = (str(task), eval_seed, int(episode["scene_seed"]))
                if key in rows:
                    raise ValueError(f"duplicate R2 scene {key}")
                rows[key] = {
                    "success": bool(episode["success"]),
                }
    efficiency = {
        "queries_per_episode": float(report["total_model_queries"]) / int(report["total_episodes"]),
        "elapsed_sec_per_episode": float(report["total_elapsed_sec"]) / int(report["total_episodes"]),
    }
    if len(report["efficiency_cells"]) != report["summary_count"]:
        raise ValueError("R2 report has incomplete efficiency cells")
    return rows, efficiency


def paired_bootstrap(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 1000):
        count = min(1000, BOOTSTRAP_RESAMPLES - start)
        index = rng.integers(0, len(delta), size=(count, len(delta)))
        samples[start : start + count] = delta[index].mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def analyze(
    fixed: dict,
    adaptive: dict,
    *,
    fixed_efficiency: dict[str, float] | None = None,
    adaptive_efficiency: dict[str, float] | None = None,
) -> dict[str, Any]:
    if set(fixed) != set(adaptive):
        raise ValueError("R2 fixed/adaptive scene identities differ")
    keys = sorted(fixed)
    fixed_success = np.asarray([fixed[key]["success"] for key in keys], dtype=np.float64)
    adaptive_success = np.asarray([adaptive[key]["success"] for key in keys], dtype=np.float64)
    delta = adaptive_success - fixed_success
    per_task = {}
    for task in sorted({key[0] for key in keys}):
        index = np.asarray([key[0] == task for key in keys])
        per_task[task] = {
            "fixed4": float(fixed_success[index].mean()),
            "adaptive": float(adaptive_success[index].mean()),
            "delta": float(delta[index].mean()),
            "episodes": int(index.sum()),
        }
    if fixed_efficiency is None:
        fixed_queries = float(np.mean([fixed[key]["cell_query_per_episode"] for key in keys]))
        fixed_elapsed = float(np.mean([fixed[key]["cell_elapsed_per_episode"] for key in keys]))
    else:
        fixed_queries = float(fixed_efficiency["queries_per_episode"])
        fixed_elapsed = float(fixed_efficiency["elapsed_sec_per_episode"])
    if adaptive_efficiency is None:
        adaptive_queries = float(np.mean([adaptive[key]["cell_query_per_episode"] for key in keys]))
        adaptive_elapsed = float(np.mean([adaptive[key]["cell_elapsed_per_episode"] for key in keys]))
    else:
        adaptive_queries = float(adaptive_efficiency["queries_per_episode"])
        adaptive_elapsed = float(adaptive_efficiency["elapsed_sec_per_episode"])
    interval = paired_bootstrap(delta)
    primary_pass = interval[0] > 0.0
    safety_pass = min(value["delta"] for value in per_task.values()) >= MAX_TASK_REGRESSION
    query_ratio = adaptive_queries / max(fixed_queries, 1e-12)
    compute_pass = query_ratio <= MAX_QUERY_RATIO
    accepted = bool(primary_pass and safety_pass and compute_pass)
    return {
        "schema_version": 1,
        "protocol": "pi05_r2_adaptive_execution_same_scene_v1",
        "episodes": len(keys),
        "fixed4_success_rate": float(fixed_success.mean()),
        "adaptive_success_rate": float(adaptive_success.mean()),
        "adaptive_minus_fixed4": float(delta.mean()),
        "paired_bootstrap_ci95": interval,
        "per_task": per_task,
        "efficiency": {
            "fixed4_queries_per_episode": fixed_queries,
            "adaptive_queries_per_episode": adaptive_queries,
            "adaptive_query_ratio": query_ratio,
            "fixed4_elapsed_sec_per_episode": fixed_elapsed,
            "adaptive_elapsed_sec_per_episode": adaptive_elapsed,
            "adaptive_wallclock_ratio": adaptive_elapsed / max(fixed_elapsed, 1e-12),
        },
        "gate": {
            "accepted": accepted,
            "paired_ci_lower_positive": primary_pass,
            "no_task_regression_below_minus_0_05": safety_pass,
            "adaptive_query_ratio_at_most_1_05": compute_pass,
            "verdict": "accepted" if accepted else "rejected",
        },
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-report", type=Path, required=True)
    parser.add_argument("--adaptive-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    fixed, fixed_efficiency = load_report(args.fixed_report)
    adaptive, adaptive_efficiency = load_report(args.adaptive_report)
    result = analyze(
        fixed,
        adaptive,
        fixed_efficiency=fixed_efficiency,
        adaptive_efficiency=adaptive_efficiency,
    )
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.gate_dir.glob("r2_gate.*"):
        stale.unlink()
    atomic_text(
        args.gate_dir / f"r2_gate.{result['gate']['verdict']}",
        f"report={args.output}\n",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

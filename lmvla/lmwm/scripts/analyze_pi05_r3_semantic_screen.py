#!/usr/bin/env python3
"""Analyze the preregistered five-arm R3 semantic prompt screen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


TARGET = "semantic_next"
CONTROLS = ("no_subtask", "generic_stage", "semantic_current", "shuffled_semantic")


def flatten(report: dict[str, Any]) -> dict[tuple[str, int, int], bool]:
    rows: dict[tuple[str, int, int], bool] = {}
    for task, task_data in report["tasks"].items():
        for cell in task_data["cells"]:
            eval_seed = int(cell["eval_seed"])
            for episode in cell["episode_outcomes"]:
                key = (str(task), eval_seed, int(episode["scene_seed"]))
                if key in rows:
                    raise ValueError(f"duplicate scene key {key}")
                rows[key] = bool(episode["success"])
    return rows


def paired_bootstrap(delta: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def analyze(reports: dict[str, dict[str, Any]], *, resamples: int, seed: int) -> dict[str, Any]:
    missing = {TARGET, *CONTROLS} - set(reports)
    if missing:
        raise ValueError(f"missing R3 conditions: {sorted(missing)}")
    rows = {condition: flatten(report) for condition, report in reports.items()}
    target_keys = set(rows[TARGET])
    for condition, values in rows.items():
        if set(values) != target_keys:
            raise ValueError(f"{condition}: scene identity differs from {TARGET}")
    keys = sorted(target_keys)
    target = np.asarray([rows[TARGET][key] for key in keys], dtype=np.float64)

    comparisons = {}
    for index, condition in enumerate(CONTROLS):
        control = np.asarray([rows[condition][key] for key in keys], dtype=np.float64)
        delta = target - control
        comparisons[condition] = {
            "target_rate": float(target.mean()),
            "control_rate": float(control.mean()),
            "delta": float(delta.mean()),
            "ci95": list(paired_bootstrap(delta, resamples=resamples, seed=seed + index)),
            "discordant_target_wins": int(np.sum(delta > 0)),
            "discordant_control_wins": int(np.sum(delta < 0)),
        }

    no_subtask = rows["no_subtask"]
    per_task = {}
    for task in sorted({key[0] for key in keys}):
        task_keys = [key for key in keys if key[0] == task]
        target_rate = float(np.mean([rows[TARGET][key] for key in task_keys]))
        baseline_rate = float(np.mean([no_subtask[key] for key in task_keys]))
        per_task[task] = {
            "semantic_next": target_rate,
            "no_subtask": baseline_rate,
            "delta": target_rate - baseline_rate,
            "episodes": len(task_keys),
        }

    primary_pass = comparisons["no_subtask"]["ci95"][0] > 0.0
    control_pass = all(comparisons[name]["delta"] > 0.0 for name in CONTROLS[1:])
    task_safety_pass = min(item["delta"] for item in per_task.values()) >= -0.05
    accepted = bool(primary_pass and control_pass and task_safety_pass)
    return {
        "protocol": "pi05_r3_semantic_screen_same_scene_v1",
        "episodes": len(keys),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
        "per_task": per_task,
        "gate": {
            "accepted": accepted,
            "primary_ci_pass": primary_pass,
            "all_control_point_estimates_pass": control_pass,
            "task_safety_pass": task_safety_pass,
            "verdict": "accepted" if accepted else "rejected",
            "implication": (
                "semantic predictor interface may proceed to specification"
                if accepted
                else "do not train an R3 semantic predictor"
            ),
        },
    }


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# pi0.5 R3 Semantic-Subtask Screen",
        "",
        f"- Verdict: `{result['gate']['verdict']}`",
        f"- Matched episodes: {result['episodes']}",
        f"- Semantic-next success: {result['comparisons']['no_subtask']['target_rate']:.3f}",
        "",
        "| Control | Control SR | Delta | Paired 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name in CONTROLS:
        row = result["comparisons"][name]
        lines.append(f"| {name} | {row['control_rate']:.3f} | {row['delta']:+.3f} | [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] |")
    lines.extend(["", f"Decision: {result['gate']['implication']}.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", nargs=2, metavar=("CONDITION", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    args = parser.parse_args()
    reports = {condition: json.loads(Path(path).read_text(encoding="utf-8")) for condition, path in args.report}
    result = analyze(reports, resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_text(args.markdown, render_markdown(result))
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    for stale in (args.gate_dir / "r3_gate.accepted", args.gate_dir / "r3_gate.rejected"):
        stale.unlink(missing_ok=True)
    atomic_text(args.gate_dir / f"r3_gate.{result['gate']['verdict']}", f"report={args.output}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

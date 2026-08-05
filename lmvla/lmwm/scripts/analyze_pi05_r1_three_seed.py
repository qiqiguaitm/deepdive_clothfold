#!/usr/bin/env python3
"""Apply the preregistered three-seed R1 recurrence-aligned gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_SEEDS = (1000, 1001, 1002)
ARMS = ("a0", "predictive", "crave", "combined")
CONTROLS = ("a0", "predictive", "crave")
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260804
MAX_TASK_REGRESSION = -0.05


def outcome_map(report: dict[str, Any]) -> dict[tuple[str, int, int], float]:
    if report.get("summary_count") != 24 or report.get("total_episodes") != 1200:
        raise ValueError("R1 reports require exactly 24 cells and 1,200 episodes")
    if len(report.get("tasks", {})) != 6:
        raise ValueError("R1 reports require six tasks")
    rows: dict[tuple[str, int, int], float] = {}
    for task, payload in report["tasks"].items():
        for cell in payload["cells"]:
            eval_seed = int(cell["eval_seed"])
            for episode in cell["episode_outcomes"]:
                key = (str(task), eval_seed, int(episode["scene_seed"]))
                if key in rows:
                    raise ValueError(f"duplicate R1 scene: {key}")
                rows[key] = float(bool(episode["success"]))
    if len(rows) != 1200:
        raise ValueError(f"R1 report has {len(rows)} unique scenes, expected 1,200")
    return rows


def hierarchical_interval(deltas: dict[int, np.ndarray], *, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    per_seed = np.empty((len(TRAIN_SEEDS), samples), dtype=np.float64)
    for seed_index, train_seed in enumerate(TRAIN_SEEDS):
        delta = deltas[train_seed]
        for start in range(0, samples, 500):
            count = min(500, samples - start)
            indices = rng.integers(0, len(delta), size=(count, len(delta)))
            per_seed[seed_index, start : start + count] = delta[indices].mean(axis=1)
    selected = rng.integers(0, len(TRAIN_SEEDS), size=(samples, len(TRAIN_SEEDS)))
    sample_index = np.arange(samples)[:, None]
    hierarchy = per_seed.T[sample_index, selected].mean(axis=1)
    return [float(value) for value in np.quantile(hierarchy, [0.025, 0.975])]


def analyze(
    reports: dict[int, dict[str, dict[str, Any]]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if tuple(sorted(reports)) != TRAIN_SEEDS:
        raise ValueError(f"R1 training seeds must be exactly {TRAIN_SEEDS}")
    maps: dict[int, dict[str, dict[tuple[str, int, int], float]]] = {}
    for train_seed in TRAIN_SEEDS:
        if tuple(sorted(reports[train_seed])) != tuple(sorted(ARMS)):
            raise ValueError(f"seed {train_seed} must contain arms {ARMS}")
        maps[train_seed] = {arm: outcome_map(reports[train_seed][arm]) for arm in ARMS}
        keys = set(maps[train_seed]["a0"])
        if any(set(rows) != keys for rows in maps[train_seed].values()):
            raise ValueError(f"seed {train_seed} arms do not share scene identities")

    comparisons: dict[str, Any] = {}
    task_effects: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for control in CONTROLS:
        paired: dict[int, np.ndarray] = {}
        seed_effects: dict[str, float] = {}
        for train_seed in TRAIN_SEEDS:
            keys = sorted(maps[train_seed]["combined"])
            paired[train_seed] = np.asarray(
                [maps[train_seed]["combined"][key] - maps[train_seed][control][key] for key in keys],
                dtype=np.float64,
            )
            seed_effects[str(train_seed)] = float(paired[train_seed].mean())
        interval = hierarchical_interval(
            paired,
            samples=bootstrap_samples,
            seed=bootstrap_seed + CONTROLS.index(control),
        )
        comparisons[control] = {
            "seed_effects": seed_effects,
            "mean_effect": float(np.mean(list(seed_effects.values()))),
            "hierarchical_paired_bootstrap_ci95": interval,
        }
        checks[f"combined_ci_lower_positive_vs_{control}"] = interval[0] > 0.0

    tasks = sorted(reports[1000]["a0"]["tasks"])
    for train_seed in TRAIN_SEEDS:
        task_effects[str(train_seed)] = {}
        for task in tasks:
            keys = [key for key in maps[train_seed]["a0"] if key[0] == task]
            task_effects[str(train_seed)][task] = float(
                np.mean([maps[train_seed]["combined"][key] - maps[train_seed]["a0"][key] for key in keys])
            )
    mean_task_effects = {
        task: float(np.mean([task_effects[str(seed)][task] for seed in TRAIN_SEEDS])) for task in tasks
    }
    checks["no_mean_task_regression_below_minus_0_05_vs_a0"] = min(mean_task_effects.values()) >= MAX_TASK_REGRESSION
    return {
        "schema_version": 1,
        "protocol": "pi05_r1_recurrence_aligned_three_seed_v1",
        "training_seeds": list(TRAIN_SEEDS),
        "arms": list(ARMS),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "levels": ["training_seed", "paired_episode"],
        },
        "comparisons": comparisons,
        "task_effects_vs_a0_by_seed": task_effects,
        "mean_task_effects_vs_a0": mean_task_effects,
        "checks": checks,
        "accepted": bool(all(checks.values())),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="SEED:ARM=REPORT.json")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    reports: dict[int, dict[str, dict[str, Any]]] = {}
    for item in args.report:
        identity, path_text = item.split("=", 1)
        seed_text, arm = identity.split(":", 1)
        train_seed = int(seed_text)
        reports.setdefault(train_seed, {})[arm] = json.loads(Path(path_text).read_text(encoding="utf-8"))
    result = analyze(reports, bootstrap_samples=args.bootstrap_samples)
    atomic_json(args.output, result)
    args.gate_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.gate_dir.glob("r1_three_seed_gate.*"):
        stale.unlink()
    verdict = "accepted" if result["accepted"] else "rejected"
    (args.gate_dir / f"r1_three_seed_gate.{verdict}").write_text(f"report={args.output}\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply the preregistered three-seed R4 replication gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analyze_pi05_r4_formal import _load_report, _validate_pairing


TRAIN_SEEDS = (1000, 1001, 1002)
ARMS = ("ordinary", "terminal_outcome", "outcome_free_crave")
CONTROLS = ("ordinary", "outcome_free_crave")
MAX_TASK_REGRESSION = -0.05
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260806


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _paired_delta(
    candidate: dict[str, dict[int, dict[int, int]]],
    control: dict[str, dict[int, dict[int, int]]],
    task: str,
    eval_seed: int,
    scene_seed: int,
) -> int:
    return candidate[task][eval_seed][scene_seed] - control[task][eval_seed][scene_seed]


def _bootstrap(
    reports: dict[int, dict[str, dict[str, dict[int, dict[int, int]]]]],
    control_arm: str,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    estimates = np.empty(samples, dtype=np.float64)
    tasks = sorted(reports[TRAIN_SEEDS[0]]["terminal_outcome"])
    for replicate in range(samples):
        seed_estimates = []
        for train_seed in rng.choice(TRAIN_SEEDS, size=len(TRAIN_SEEDS), replace=True):
            train_seed = int(train_seed)
            candidate = reports[train_seed]["terminal_outcome"]
            control = reports[train_seed][control_arm]
            task_estimates = []
            for task in rng.choice(tasks, size=len(tasks), replace=True):
                eval_seeds = sorted(candidate[str(task)])
                cell_estimates = []
                for eval_seed in rng.choice(eval_seeds, size=len(eval_seeds), replace=True):
                    eval_seed = int(eval_seed)
                    scenes = np.asarray(sorted(candidate[str(task)][eval_seed]), dtype=np.int64)
                    sampled_scenes = rng.choice(scenes, size=len(scenes), replace=True)
                    cell_estimates.append(
                        float(
                            np.mean(
                                [
                                    _paired_delta(
                                        candidate,
                                        control,
                                        str(task),
                                        eval_seed,
                                        int(scene_seed),
                                    )
                                    for scene_seed in sampled_scenes
                                ]
                            )
                        )
                    )
                task_estimates.append(float(np.mean(cell_estimates)))
            seed_estimates.append(float(np.mean(task_estimates)))
        estimates[replicate] = float(np.mean(seed_estimates))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def analyze(
    report_paths: dict[tuple[int, str], Path],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    expected = {(seed, arm) for seed in TRAIN_SEEDS for arm in ARMS}
    if set(report_paths) != expected:
        raise ValueError(
            f"reports must be exactly the nine seed/arm cells; missing={sorted(expected-set(report_paths))}, "
            f"extra={sorted(set(report_paths)-expected)}"
        )

    cells: dict[int, dict[str, dict]] = {}
    macros: dict[int, dict[str, float]] = {}
    for seed in TRAIN_SEEDS:
        cells[seed] = {}
        macros[seed] = {}
        for arm in ARMS:
            cells[seed][arm], macros[seed][arm] = _load_report(report_paths[(seed, arm)])
        for control in CONTROLS:
            _validate_pairing(cells[seed]["terminal_outcome"], cells[seed][control], control)

    comparisons = {}
    rng = np.random.default_rng(bootstrap_seed)
    for control in CONTROLS:
        seed_effects = {
            str(seed): macros[seed]["terminal_outcome"] - macros[seed][control]
            for seed in TRAIN_SEEDS
        }
        seed_task_effects = {}
        for seed in TRAIN_SEEDS:
            candidate = cells[seed]["terminal_outcome"]
            baseline = cells[seed][control]
            seed_task_effects[str(seed)] = {
                task: float(
                    np.mean(
                        [
                            _paired_delta(candidate, baseline, task, eval_seed, scene_seed)
                            for eval_seed in candidate[task]
                            for scene_seed in candidate[task][eval_seed]
                        ]
                    )
                )
                for task in sorted(candidate)
            }
        task_effects = {
            task: float(np.mean([seed_task_effects[str(seed)][task] for seed in TRAIN_SEEDS]))
            for task in sorted(cells[TRAIN_SEEDS[0]]["terminal_outcome"])
        }
        interval = _bootstrap(
            cells,
            control,
            samples=bootstrap_samples,
            rng=rng,
        )
        comparisons[control] = {
            "seed_effects": seed_effects,
            "mean_seed_effect": float(np.mean(list(seed_effects.values()))),
            "seed_task_effects": seed_task_effects,
            "task_effects": task_effects,
            "hierarchical_paired_bootstrap_95": interval,
            "positive_95ci": interval[0] > 0.0,
            "no_seed_task_regression_below_minus_0_05": min(
                value
                for task_values in seed_task_effects.values()
                for value in task_values.values()
            )
            >= MAX_TASK_REGRESSION,
        }

    checks = {
        f"positive_95ci_vs_{control}": comparisons[control]["positive_95ci"]
        for control in CONTROLS
    }
    checks.update(
        {
            f"task_safety_vs_{control}": comparisons[control][
                "no_seed_task_regression_below_minus_0_05"
            ]
            for control in CONTROLS
        }
    )
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_three_seed_replication_v1",
        "training_seeds": list(TRAIN_SEEDS),
        "checkpoint_step": 5000,
        "macros": {str(seed): macros[seed] for seed in TRAIN_SEEDS},
        "comparisons": comparisons,
        "bootstrap": {
            "levels": ["training_seed", "task", "evaluation_seed", "paired_scene"],
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "checks": checks,
        "accepted": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="SEED:ARM=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    parser.add_argument("--rejected-marker", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    reports = {}
    for item in args.report:
        key, path = item.split("=", 1)
        seed, arm = key.split(":", 1)
        report_key = (int(seed), arm)
        if report_key in reports:
            raise ValueError(f"duplicate report: {report_key}")
        reports[report_key] = Path(path)
    result = analyze(reports, bootstrap_samples=args.bootstrap_samples)
    _atomic_json(args.output, result)
    marker = args.accepted_marker if result["accepted"] else args.rejected_marker
    other = args.rejected_marker if result["accepted"] else args.accepted_marker
    other.unlink(missing_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"accepted={str(result['accepted']).lower()}\nresult={args.output.resolve()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

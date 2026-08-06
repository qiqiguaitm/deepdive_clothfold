#!/usr/bin/env python3
"""Apply the frozen matched-training-seed P3 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import atomic_write_text, outcome_map  # noqa: E402
from analyze_pi05_predictive_adapter_p2 import validate_report  # noqa: E402


TRAIN_SEEDS = (1000, 1001, 1002)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260806


def _delta(candidate: dict, baseline: dict, task: str) -> np.ndarray:
    candidate_map = outcome_map(candidate, task)
    baseline_map = outcome_map(baseline, task)
    if set(candidate_map) != set(baseline_map):
        raise ValueError(f"candidate and A0 scene keys differ for task {task}")
    keys = sorted(candidate_map)
    if len(keys) != 200:
        raise ValueError(f"expected 200 paired episodes for task {task}, got {len(keys)}")
    return np.asarray(
        [int(candidate_map[key]) - int(baseline_map[key]) for key in keys],
        dtype=np.float64,
    )


def analyze(
    baselines: dict[int, dict],
    candidates: dict[int, dict],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    if tuple(sorted(baselines)) != TRAIN_SEEDS or tuple(sorted(candidates)) != TRAIN_SEEDS:
        raise ValueError(f"baseline and candidate seeds must both be exactly {TRAIN_SEEDS}")
    for seed in TRAIN_SEEDS:
        validate_report(f"a0_seed{seed}", baselines[seed])
        validate_report(f"candidate_seed{seed}", candidates[seed])

    tasks = sorted(baselines[1000]["tasks"])
    if len(tasks) != 6:
        raise ValueError("expected exactly six equally weighted tasks")
    paired: dict[int, dict[str, np.ndarray]] = {}
    for seed in TRAIN_SEEDS:
        if sorted(baselines[seed]["tasks"]) != tasks or sorted(candidates[seed]["tasks"]) != tasks:
            raise ValueError(f"task set mismatch at training seed {seed}")
        paired[seed] = {
            task: _delta(candidates[seed], baselines[seed], task) for task in tasks
        }

    task_effects = {
        str(seed): {task: float(values.mean()) for task, values in paired[seed].items()}
        for seed in TRAIN_SEEDS
    }
    seed_effects = {
        str(seed): float(np.mean(list(task_effects[str(seed)].values())))
        for seed in TRAIN_SEEDS
    }

    rng = np.random.default_rng(bootstrap_seed)
    per_seed = np.zeros((len(TRAIN_SEEDS), bootstrap_samples), dtype=np.float64)
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        for values in paired[seed].values():
            indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            per_seed[seed_index] += values[indices].mean(axis=1) / len(tasks)
    sampled_seeds = rng.integers(
        0, len(TRAIN_SEEDS), size=(bootstrap_samples, len(TRAIN_SEEDS))
    )
    rows = np.arange(bootstrap_samples)[:, None]
    draws = per_seed.T[rows, sampled_seeds].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    positive = float(lower) > 0.0
    task_safe = min(value for row in task_effects.values() for value in row.values()) >= -0.05
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_p3_matched_seed_gate_v1",
        "complete": True,
        "training_seeds": list(TRAIN_SEEDS),
        "tasks": tasks,
        "candidate_macro_success_rates": {
            str(seed): candidates[seed]["macro_success_rate"] for seed in TRAIN_SEEDS
        },
        "a0_macro_success_rates": {
            str(seed): baselines[seed]["macro_success_rate"] for seed in TRAIN_SEEDS
        },
        "seed_effects_candidate_minus_a0": seed_effects,
        "task_effects_candidate_minus_a0": task_effects,
        "mean_matched_seed_effect": float(np.mean(list(seed_effects.values()))),
        "hierarchical_paired_bootstrap": {
            "hierarchy": "resample training seeds, then paired episodes within each equally weighted task",
            "samples": bootstrap_samples,
            "random_seed": bootstrap_seed,
            "ci95": [float(lower), float(upper)],
        },
        "checks": {
            "three_independently_matched_training_seeds": True,
            "exact_episode_pairing": True,
            "hierarchical_95ci_excludes_zero_positive": positive,
            "no_seed_task_effect_below_minus_5_points": task_safe,
        },
        "accepted": positive,
        "task_safe": task_safe,
    }


def _load_seed_paths(values: list[str]) -> dict[int, dict]:
    reports = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in reports:
            raise ValueError(f"duplicate training seed {seed}")
        reports[seed] = json.loads(Path(path_text).read_text())
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--candidate", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        _load_seed_paths(args.a0),
        _load_seed_paths(args.candidate),
        bootstrap_samples=args.bootstrap_samples,
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        args.marker,
        f"validated=true\naccepted={str(result['accepted']).lower()}\n"
        f"task_safe={str(result['task_safe']).lower()}\nreport={args.output}\n",
    )
    print(json.dumps({"accepted": result["accepted"], "task_safe": result["task_safe"]}, indent=2))


if __name__ == "__main__":
    main()

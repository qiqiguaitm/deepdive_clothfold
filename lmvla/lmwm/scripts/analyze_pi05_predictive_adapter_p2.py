#!/usr/bin/env python3
"""Apply the preregistered hierarchical P2 predictive-adapter gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import atomic_write_text, outcome_map  # noqa: E402


TRAIN_SEEDS = (1000, 1001, 1002)
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_SAMPLES = 20_000


def validate_report(name: str, report: dict) -> None:
    if report.get("summary_count") != 24:
        raise ValueError(f"{name}: expected 24 cells, got {report.get('summary_count')}")
    if report.get("total_episodes") != 1200:
        raise ValueError(
            f"{name}: expected 1200 episodes, got {report.get('total_episodes')}"
        )
    if len(report.get("tasks", {})) != 6:
        raise ValueError(f"{name}: expected six tasks")


def _paired_vectors(candidate: dict, baseline: dict, task: str) -> tuple[np.ndarray, np.ndarray]:
    candidate_map = outcome_map(candidate, task)
    baseline_map = outcome_map(baseline, task)
    if set(candidate_map) != set(baseline_map):
        raise ValueError(f"candidate and A0 scene keys differ for task {task}")
    keys = sorted(candidate_map)
    return (
        np.asarray([candidate_map[key] for key in keys], dtype=np.float64),
        np.asarray([baseline_map[key] for key in keys], dtype=np.float64),
    )


def apply_gate(
    baseline: dict,
    candidates: dict[int, dict],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    validate_report("a0_seed1000", baseline)
    if tuple(sorted(candidates)) != TRAIN_SEEDS:
        raise ValueError(f"candidate training seeds must be exactly {TRAIN_SEEDS}")
    for seed, report in candidates.items():
        validate_report(f"candidate_seed{seed}", report)
        if set(outcome_map(report)) != set(outcome_map(baseline)):
            raise ValueError(f"candidate seed {seed} does not exactly match A0 scene keys")

    tasks = sorted(baseline["tasks"])
    seed_effects = {
        str(seed): report["macro_success_rate"] - baseline["macro_success_rate"]
        for seed, report in candidates.items()
    }
    task_effects = {
        str(seed): {
            task: report["tasks"][task]["mean_success_rate"]
            - baseline["tasks"][task]["mean_success_rate"]
            for task in tasks
        }
        for seed, report in candidates.items()
    }

    rng = np.random.default_rng(bootstrap_seed)
    per_seed_bootstrap = np.zeros((len(TRAIN_SEEDS), bootstrap_samples), dtype=np.float64)
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        for task in tasks:
            candidate_values, baseline_values = _paired_vectors(candidates[seed], baseline, task)
            indices = rng.integers(
                0,
                len(candidate_values),
                size=(bootstrap_samples, len(candidate_values)),
            )
            paired_delta = candidate_values - baseline_values
            per_seed_bootstrap[seed_index] += paired_delta[indices].mean(axis=1) / len(tasks)

    selected_seeds = rng.integers(
        0, len(TRAIN_SEEDS), size=(bootstrap_samples, len(TRAIN_SEEDS))
    )
    replicate_index = np.arange(bootstrap_samples)[:, None]
    hierarchical = per_seed_bootstrap.T[replicate_index, selected_seeds].mean(axis=1)
    lower, upper = np.quantile(hierarchical, [0.025, 0.975])
    point_estimate = float(np.mean(list(seed_effects.values())))
    checks = {
        "all_three_training_seeds_present": tuple(sorted(candidates)) == TRAIN_SEEDS,
        "exact_scene_pairing": True,
        "hierarchical_95ci_excludes_zero_positive": float(lower) > 0.0,
    }
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_action_adapter_p2_hierarchical_gate_v1",
        "complete": True,
        "baseline": {
            "training_seed": 1000,
            "macro_success_rate": baseline["macro_success_rate"],
        },
        "candidate_training_seeds": list(TRAIN_SEEDS),
        "candidate_macro_success_rates": {
            str(seed): candidates[seed]["macro_success_rate"] for seed in TRAIN_SEEDS
        },
        "seed_effects_vs_a0": seed_effects,
        "task_effects_vs_a0": task_effects,
        "mean_seed_effect_vs_a0": point_estimate,
        "hierarchical_paired_bootstrap": {
            "unit_levels": ["candidate_training_seed", "paired_episode_within_task"],
            "samples": bootstrap_samples,
            "random_seed": bootstrap_seed,
            "ci95": [float(lower), float(upper)],
        },
        "checks": checks,
        "accepted": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = {}
    for value in args.candidate:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in candidates:
            raise ValueError(f"duplicate candidate seed: {seed}")
        candidates[seed] = json.loads(Path(path_text).read_text())
    result = apply_gate(
        json.loads(args.a0.read_text()),
        candidates,
        bootstrap_samples=args.bootstrap_samples,
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": result["checks"], "accepted": result["accepted"]}, indent=2))


if __name__ == "__main__":
    main()

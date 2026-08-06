#!/usr/bin/env python3
"""Analyze the frozen three-seed P4 inference-intervention family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import (  # noqa: E402
    atomic_write_text,
    exact_mcnemar,
    holm_adjust,
    outcome_map,
)
from analyze_pi05_predictive_adapter_p2 import validate_report  # noqa: E402


TRAIN_SEEDS = (1000, 1001, 1002)
CONTROLS = ("shuffled", "zero_gate", "masked")
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260806


def _paired_delta(normal: dict, control: dict, task: str) -> np.ndarray:
    normal_map = outcome_map(normal, task)
    control_map = outcome_map(control, task)
    if set(normal_map) != set(control_map):
        raise ValueError(f"normal and control scene keys differ for task {task}")
    keys = sorted(normal_map)
    if len(keys) != 200:
        raise ValueError(f"expected 200 paired episodes for task {task}, got {len(keys)}")
    return np.asarray(
        [int(normal_map[key]) - int(control_map[key]) for key in keys],
        dtype=np.float64,
    )


def _pooled_maps(reports: dict[int, dict]) -> dict[tuple[int, str, int, int], bool]:
    pooled = {}
    for seed in TRAIN_SEEDS:
        for key, value in outcome_map(reports[seed]).items():
            pooled[(seed, *key)] = value
    return pooled


def analyze(
    normal: dict[int, dict],
    controls: dict[str, dict[int, dict]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    if tuple(sorted(normal)) != TRAIN_SEEDS:
        raise ValueError(f"normal seeds must be exactly {TRAIN_SEEDS}")
    if tuple(controls) != CONTROLS:
        raise ValueError(f"controls must be supplied in frozen order {CONTROLS}")
    for condition in CONTROLS:
        if tuple(sorted(controls[condition])) != TRAIN_SEEDS:
            raise ValueError(f"{condition} seeds must be exactly {TRAIN_SEEDS}")
    for seed in TRAIN_SEEDS:
        validate_report(f"normal_seed{seed}", normal[seed])
        for condition in CONTROLS:
            validate_report(f"{condition}_seed{seed}", controls[condition][seed])

    tasks = sorted(normal[1000]["tasks"])
    rng = np.random.default_rng(bootstrap_seed)
    comparisons = {}
    pooled_tests = []
    normal_pooled = _pooled_maps(normal)
    for comparison_index, condition in enumerate(CONTROLS):
        paired = {
            seed: {
                task: _paired_delta(normal[seed], controls[condition][seed], task)
                for task in tasks
            }
            for seed in TRAIN_SEEDS
        }
        task_effects = {
            str(seed): {task: float(values.mean()) for task, values in paired[seed].items()}
            for seed in TRAIN_SEEDS
        }
        seed_effects = {
            str(seed): float(np.mean(list(task_effects[str(seed)].values())))
            for seed in TRAIN_SEEDS
        }
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
        pooled_test = exact_mcnemar(normal_pooled, _pooled_maps(controls[condition]))
        pooled_test["comparison"] = f"normal_minus_{condition}"
        pooled_tests.append(pooled_test)
        comparisons[condition] = {
            "normal_macro_success_rates": {
                str(seed): normal[seed]["macro_success_rate"] for seed in TRAIN_SEEDS
            },
            "control_macro_success_rates": {
                str(seed): controls[condition][seed]["macro_success_rate"] for seed in TRAIN_SEEDS
            },
            "seed_effects_normal_minus_control": seed_effects,
            "task_effects_normal_minus_control": task_effects,
            "mean_seed_effect": float(np.mean(list(seed_effects.values()))),
            "hierarchical_paired_bootstrap": {
                "samples": bootstrap_samples,
                "random_seed": bootstrap_seed,
                "comparison_order_index": comparison_index,
                "ci95": [float(lower), float(upper)],
            },
        }
    holm_adjust(pooled_tests)
    tests_by_condition = {
        row["comparison"].removeprefix("normal_minus_"): row for row in pooled_tests
    }
    for condition, row in comparisons.items():
        test = tests_by_condition[condition]
        row["pooled_exact_mcnemar"] = test
        row["accepted"] = (
            row["hierarchical_paired_bootstrap"]["ci95"][0] > 0.0
            and test["holm_adjusted_p"] < 0.05
        )
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_p4_three_seed_interventions_v1",
        "complete": True,
        "training_seeds": list(TRAIN_SEEDS),
        "tasks": tasks,
        "holm_family": "three pooled paired tests: normal minus shuffled, zero_gate, and masked",
        "comparisons": comparisons,
        "claim_gates": {
            "content_specific_causality": comparisons["shuffled"]["accepted"],
            "route_necessity": comparisons["zero_gate"]["accepted"],
            "action_conditioning_use": comparisons["masked"]["accepted"],
        },
    }


def _load(values: list[str]) -> dict[int, dict]:
    reports = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in reports:
            raise ValueError(f"duplicate seed {seed}")
        reports[seed] = json.loads(Path(path_text).read_text())
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", action="append", required=True, help="SEED=REPORT.json")
    for condition in CONTROLS:
        parser.add_argument(f"--{condition.replace('_', '-')}", action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    controls = {
        "shuffled": _load(args.shuffled),
        "zero_gate": _load(args.zero_gate),
        "masked": _load(args.masked),
    }
    result = analyze(
        _load(args.normal), controls, bootstrap_samples=args.bootstrap_samples
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    gates = result["claim_gates"]
    atomic_write_text(
        args.marker,
        "validated=true\n" + "\n".join(f"{key}={str(value).lower()}" for key, value in gates.items()) + "\n",
    )
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()

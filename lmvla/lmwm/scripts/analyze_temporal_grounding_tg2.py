#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import atomic_write_text  # noqa: E402
from analyze_temporal_grounding_tg1a import EVAL_SEEDS, load_condition  # noqa: E402


ARMS = ("future_off", "fixed_endpoint", "raw_milestone")
TRAIN_SEEDS = (1000, 1001, 1002)
COMPARISONS = (
    ("fixed_endpoint", "future_off"),
    ("raw_milestone", "future_off"),
    ("fixed_endpoint", "raw_milestone"),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260809


def paired_hierarchy(candidate: dict, baseline: dict) -> dict[str, dict[int, np.ndarray]]:
    if set(candidate["records"]) != set(baseline["records"]):
        raise ValueError("TG2 arm outcomes are not exactly paired")
    paired = {}
    for task in candidate["tasks"]:
        paired[task] = {}
        for eval_seed in EVAL_SEEDS:
            keys = sorted(
                key for key in candidate["records"] if key[0] == eval_seed and key[1] == task
            )
            paired[task][eval_seed] = np.asarray(
                [int(candidate["records"][key]) - int(baseline["records"][key]) for key in keys],
                dtype=np.float64,
            )
    return paired


def hierarchical_training_seed_ci(
    paired: dict[int, dict[str, dict[int, np.ndarray]]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    training_seeds = tuple(sorted(paired))
    tasks = tuple(sorted(next(iter(paired.values()))))
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        sampled_training = rng.integers(0, len(training_seeds), size=len(training_seeds))
        values = []
        for training_index in sampled_training:
            train_seed = training_seeds[int(training_index)]
            sampled_tasks = rng.integers(0, len(tasks), size=len(tasks))
            for task_index in sampled_tasks:
                task = tasks[int(task_index)]
                sampled_eval = rng.integers(0, len(EVAL_SEEDS), size=len(EVAL_SEEDS))
                for eval_index in sampled_eval:
                    episodes = paired[train_seed][task][EVAL_SEEDS[int(eval_index)]]
                    sampled_episodes = rng.integers(0, len(episodes), size=len(episodes))
                    values.append(float(episodes[sampled_episodes].mean()))
        draws[draw_index] = float(np.mean(values))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def analyze(
    reports: dict[str, dict[int, dict]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    if tuple(reports) != ARMS:
        raise ValueError(f"TG2 arms must be supplied in frozen order {ARMS}")
    for arm in ARMS:
        if tuple(sorted(reports[arm])) != TRAIN_SEEDS:
            raise ValueError(f"TG2 {arm} must contain training seeds {TRAIN_SEEDS}")
    comparisons = {}
    for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
        paired = {
            seed: paired_hierarchy(reports[candidate][seed], reports[baseline][seed])
            for seed in TRAIN_SEEDS
        }
        seed_task_effects = {
            str(seed): {
                task: float(np.mean([values.mean() for values in eval_map.values()]))
                for task, eval_map in paired[seed].items()
            }
            for seed in TRAIN_SEEDS
        }
        seed_effects = {
            str(seed): float(np.mean(list(seed_task_effects[str(seed)].values())))
            for seed in TRAIN_SEEDS
        }
        lower, upper = hierarchical_training_seed_ci(
            paired,
            samples=bootstrap_samples,
            seed=bootstrap_seed + comparison_index,
        )
        task_safe = min(
            effect
            for task_effects in seed_task_effects.values()
            for effect in task_effects.values()
        ) >= -0.05
        name = f"{candidate}_minus_{baseline}"
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "seed_effects": seed_effects,
            "seed_task_effects": seed_task_effects,
            "equal_training_seed_mean": float(np.mean(list(seed_effects.values()))),
            "hierarchical_paired_bootstrap": {
                "levels": [
                    "training_seed",
                    "task_within_training_seed",
                    "eval_seed_within_task",
                    "paired_episode",
                ],
                "samples": bootstrap_samples,
                "random_seed": bootstrap_seed + comparison_index,
                "ci95": [lower, upper],
            },
            "task_safe": task_safe,
            "accepted": lower > 0.0 and task_safe,
        }

    fixed_utility = comparisons["fixed_endpoint_minus_future_off"]
    raw_utility = comparisons["raw_milestone_minus_future_off"]
    horizon = comparisons["fixed_endpoint_minus_raw_milestone"]
    fixed_raw_unresolved = horizon["hierarchical_paired_bootstrap"]["ci95"][0] <= 0.0 <= horizon[
        "hierarchical_paired_bootstrap"
    ]["ci95"][1]
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg2_execution_aligned_matched_matrix_v1",
        "complete": True,
        "training_seeds": list(TRAIN_SEEDS),
        "arms": list(ARMS),
        "tasks": list(reports["future_off"][1000]["tasks"]),
        "H": 50,
        "E": 50,
        "fixed_final_step": 20000,
        "comparisons": comparisons,
        "claim_gates": {
            "fixed_endpoint_utility": fixed_utility["accepted"],
            "raw_milestone_utility": raw_utility["accepted"],
            "target_horizon_effect": horizon["accepted"],
            "task_safety_fixed_vs_off": fixed_utility["task_safe"],
            "task_safety_fixed_vs_raw": horizon["task_safe"],
        },
        "stop_decision": {
            "both_active_targets_fail_utility_gate": not fixed_utility["accepted"]
            and not raw_utility["accepted"],
            "fixed_and_raw_statistically_unresolved": fixed_raw_unresolved,
            "tg3_authorized": horizon["accepted"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", action="append", required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    scene_bytes = args.scene_manifest.read_bytes()
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    scene_manifest = json.loads(scene_bytes)

    reports = {}
    for arm in ARMS:
        reports[arm] = {}
        for value in getattr(args, arm):
            seed_text, root_text = value.split("=", 1)
            seed = int(seed_text)
            if seed in reports[arm]:
                raise ValueError(f"Duplicate {arm} training seed {seed}")
            reports[arm][seed] = load_condition(Path(root_text), scene_manifest, scene_hash)
    result = analyze(reports, bootstrap_samples=args.bootstrap_samples)
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        args.marker,
        "validated=true\n"
        + "\n".join(
            f"{key}={str(value).lower()}" for key, value in result["claim_gates"].items()
        )
        + "\n"
        + "\n".join(
            f"{key}={str(value).lower()}" for key, value in result["stop_decision"].items()
        )
        + "\n",
    )
    print(json.dumps({"claim_gates": result["claim_gates"], "stop_decision": result["stop_decision"]}, indent=2))


if __name__ == "__main__":
    main()

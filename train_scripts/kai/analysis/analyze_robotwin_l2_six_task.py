#!/usr/bin/env python3
"""Analyze complete six-task LaWAM hint interventions with paired outcomes."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


TASK_DURATIONS = {
    "beat_block_hammer": 114.1,
    "handover_block": 284.3,
    "stack_blocks_two": 313.2,
    "blocks_ranking_rgb": 459.2,
    "blocks_ranking_size": 459.7,
    "stack_blocks_three": 470.7,
}
METHODS = ("absolute", "residual", "combo")
CONDITIONS = ("zero", "cross_task", "within_task_shuffle")


def permutation_interaction(
    durations: np.ndarray, deltas: np.ndarray
) -> dict[str, float]:
    centered = durations - durations.mean()
    slope = float(np.dot(centered, deltas) / np.dot(centered, centered))
    duration_ranks = rankdata(durations)
    delta_ranks = rankdata(deltas)
    rho = float(np.corrcoef(duration_ranks, delta_ranks)[0, 1])
    perm_slopes = []
    perm_rhos = []
    for permutation in itertools.permutations(deltas.tolist()):
        permuted = np.asarray(permutation)
        perm_slopes.append(float(np.dot(centered, permuted) / np.dot(centered, centered)))
        perm_rhos.append(
            float(np.corrcoef(duration_ranks, rankdata(permuted))[0, 1])
        )
    slope_p = sum(abs(value) >= abs(slope) - 1e-12 for value in perm_slopes) / len(
        perm_slopes
    )
    rho_p = sum(abs(value) >= abs(rho) - 1e-12 for value in perm_rhos) / len(
        perm_rhos
    )
    return {
        "slope_pp_per_100_frames": 100.0 * slope,
        "slope_permutation_p": slope_p,
        "spearman_rho": rho,
        "spearman_permutation_p": rho_p,
    }


def holm(values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * value))
        adjusted[label] = running
    return adjusted


def roots(base: Path, method: str) -> dict[str, list[str]]:
    prefix = base / f"rt_all6_v2_{method}"
    return {
        "correct": [str(base / f"rt_all6_v2_{method}_seed2026_unseen")],
        "zero": [str(Path(f"{prefix}_zerohint_seed2026_strict_unseen"))],
        "cross_task": [str(Path(f"{prefix}_othertask_seed2026_strict_unseen"))],
        "within_task_shuffle": [
            str(Path(f"{prefix}_shuffledhint_seed2026_strict_unseen")),
            str(Path(f"{prefix}_instanceshuffle_seed2026_strict_unseen")),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    script_dir = args.repo / "lmvla/lmwm/scripts"
    sys.path.insert(0, str(script_dir))
    from rt_causal_intervention_analysis import (  # noqa: PLC0415
        add_holm_adjustment,
        compare,
        load_episodes,
    )

    base = args.repo / "lmvla/lawam/results/eval_runs/robotwin"
    results: dict[str, dict[str, object]] = {}
    flat_rows = {}
    interactions = {}
    interaction_tests: list[tuple[str, float]] = []

    for method in METHODS:
        method_roots = roots(base, method)
        correct = load_episodes(method_roots["correct"])
        if len(correct) != 1200:
            raise SystemExit(f"{method} correct episodes incomplete: {len(correct)}/1200")
        results[method] = {}
        for condition in CONDITIONS:
            control = load_episodes(method_roots[condition])
            if len(control) != 1200:
                raise SystemExit(
                    f"{method} {condition} episodes incomplete: {len(control)}/1200"
                )
            shared = set(correct) & set(control)
            if len(shared) != 1200:
                raise SystemExit(
                    f"{method} {condition} paired cohort incomplete: {len(shared)}/1200"
                )
            rows = compare(correct, control, shared)
            results[method][condition] = {
                "roots": method_roots[condition],
                "rows": rows,
            }
            flat_rows[f"{method}:{condition}"] = rows
            by_task = {row["task"]: row for row in rows if row["task"] != "POOLED"}
            task_order = sorted(TASK_DURATIONS)
            interaction = permutation_interaction(
                np.asarray([TASK_DURATIONS[task] for task in task_order]),
                np.asarray([float(by_task[task]["delta_pp"]) for task in task_order]),
            )
            label = f"{method}:{condition}"
            interactions[label] = interaction
            interaction_tests.append((label, interaction["slope_permutation_p"]))

    add_holm_adjustment(flat_rows)
    adjusted = holm(interaction_tests)
    for label, value in adjusted.items():
        interactions[label]["slope_permutation_p_holm"] = value

    report = {
        "protocol": {
            "methods": list(METHODS),
            "conditions": list(CONDITIONS),
            "paired_episodes_per_comparison": 1200,
            "duration_source": "lmvla/lmwm/docs/ANALYSIS_lmvla_task_regime_2026-08-01.md",
            "within_task_control": (
                "spatial token shuffle on the original three-task panel; "
                "same-task foreign-episode hint on the three added tasks"
            ),
        },
        "task_mean_demo_frames": TASK_DURATIONS,
        "comparisons": results,
        "duration_interactions": interactions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "comparisons": 9}, sort_keys=True))


if __name__ == "__main__":
    main()

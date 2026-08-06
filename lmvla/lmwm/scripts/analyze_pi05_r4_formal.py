#!/usr/bin/env python3
"""Apply the preregistered R4 seed-1000 matched-arm gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


EXPECTED_TASKS = 6
EXPECTED_SEEDS = 4
EXPECTED_EPISODES_PER_CELL = 50
MAX_TASK_REGRESSION = -0.05


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _load_report(path: Path) -> tuple[dict[str, dict[int, dict[int, int]]], float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("summary_count") != EXPECTED_TASKS * EXPECTED_SEEDS:
        raise ValueError(f"{path}: expected 24 cells")
    if report.get("task_count") != EXPECTED_TASKS:
        raise ValueError(f"{path}: expected six tasks")
    if report.get("total_episodes") != (
        EXPECTED_TASKS * EXPECTED_SEEDS * EXPECTED_EPISODES_PER_CELL
    ):
        raise ValueError(f"{path}: expected 1200 episodes")

    cells: dict[str, dict[int, dict[int, int]]] = {}
    for task, task_payload in sorted(report["tasks"].items()):
        task_cells: dict[int, dict[int, int]] = {}
        for cell in task_payload["cells"]:
            seed = int(cell["eval_seed"])
            outcomes = {
                int(item["scene_seed"]): int(bool(item["success"]))
                for item in cell["episode_outcomes"]
            }
            if len(outcomes) != EXPECTED_EPISODES_PER_CELL:
                raise ValueError(f"{path}: {task}/seed{seed} is not a 50-scene cell")
            if seed in task_cells:
                raise ValueError(f"{path}: duplicate {task}/seed{seed}")
            task_cells[seed] = outcomes
        if sorted(task_cells) != list(range(EXPECTED_SEEDS)):
            raise ValueError(f"{path}: {task} does not contain eval seeds 0..3")
        cells[task] = task_cells
    if len(cells) != EXPECTED_TASKS:
        raise ValueError(f"{path}: expected six unique tasks")

    task_rates = [
        np.mean([value for seed in task_cells.values() for value in seed.values()])
        for task_cells in cells.values()
    ]
    macro = float(np.mean(task_rates))
    if not np.isclose(macro, float(report["macro_success_rate"]), atol=1e-12):
        raise ValueError(f"{path}: macro_success_rate does not match episode outcomes")
    return cells, macro


def _validate_pairing(
    terminal: dict[str, dict[int, dict[int, int]]],
    control: dict[str, dict[int, dict[int, int]]],
    label: str,
) -> None:
    if terminal.keys() != control.keys():
        raise ValueError(f"task mismatch against {label}")
    for task in terminal:
        if terminal[task].keys() != control[task].keys():
            raise ValueError(f"eval-seed mismatch for {task} against {label}")
        for seed in terminal[task]:
            if terminal[task][seed].keys() != control[task][seed].keys():
                raise ValueError(
                    f"scene-seed mismatch for {task}/seed{seed} against {label}"
                )


def _task_deltas(
    terminal: dict[str, dict[int, dict[int, int]]],
    control: dict[str, dict[int, dict[int, int]]],
) -> dict[str, float]:
    result = {}
    for task in terminal:
        paired = [
            terminal[task][seed][scene] - control[task][seed][scene]
            for seed in terminal[task]
            for scene in terminal[task][seed]
        ]
        result[task] = float(np.mean(paired))
    return result


def _hierarchical_bootstrap(
    terminal: dict[str, dict[int, dict[int, int]]],
    control: dict[str, dict[int, dict[int, int]]],
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    tasks = sorted(terminal)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        task_estimates = []
        for task in rng.choice(tasks, size=len(tasks), replace=True):
            seeds = sorted(terminal[task])
            cell_estimates = []
            for seed in rng.choice(seeds, size=len(seeds), replace=True):
                scenes = np.asarray(sorted(terminal[task][int(seed)]), dtype=np.int64)
                sampled = rng.choice(scenes, size=len(scenes), replace=True)
                differences = [
                    terminal[task][int(seed)][int(scene)]
                    - control[task][int(seed)][int(scene)]
                    for scene in sampled
                ]
                cell_estimates.append(float(np.mean(differences)))
            task_estimates.append(float(np.mean(cell_estimates)))
        estimates[index] = float(np.mean(task_estimates))
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def analyze(
    ordinary_path: Path,
    terminal_path: Path,
    crave_path: Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260805,
) -> dict:
    ordinary, ordinary_macro = _load_report(ordinary_path)
    terminal, terminal_macro = _load_report(terminal_path)
    crave, crave_macro = _load_report(crave_path)
    _validate_pairing(terminal, ordinary, "ordinary")
    _validate_pairing(terminal, crave, "outcome_free_crave")

    rng = np.random.default_rng(bootstrap_seed)
    comparisons = {}
    for label, control, control_macro in (
        ("ordinary", ordinary, ordinary_macro),
        ("outcome_free_crave", crave, crave_macro),
    ):
        task_deltas = _task_deltas(terminal, control)
        comparisons[label] = {
            "control_macro_success_rate": control_macro,
            "terminal_minus_control_macro": terminal_macro - control_macro,
            "task_deltas": task_deltas,
            "no_task_regression_below_minus_0_05": (
                min(task_deltas.values()) >= MAX_TASK_REGRESSION
            ),
            "hierarchical_paired_bootstrap_95": _hierarchical_bootstrap(
                terminal,
                control,
                samples=bootstrap_samples,
                rng=rng,
            ),
        }

    accepted = all(
        item["terminal_minus_control_macro"] > 0.0
        and item["no_task_regression_below_minus_0_05"]
        for item in comparisons.values()
    )
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_formal_eval_protocol_v1",
        "training_seed": 1000,
        "checkpoint_step": 5000,
        "terminal_macro_success_rate": terminal_macro,
        "comparisons": comparisons,
        "accepted": accepted,
        "gate": (
            "terminal_outcome macro must exceed both matched controls and no "
            "task may regress by more than 5 percentage points against either"
        ),
        "bootstrap": {
            "method": "paired task/seed/episode hierarchical bootstrap",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "intervals_are_descriptive_not_gate_conditions": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--terminal-outcome", type=Path, required=True)
    parser.add_argument("--outcome-free-crave", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    parser.add_argument("--rejected-marker", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(
        args.ordinary,
        args.terminal_outcome,
        args.outcome_free_crave,
        bootstrap_samples=args.bootstrap_samples,
    )
    _atomic_json(args.output, result)
    marker = args.accepted_marker if result["accepted"] else args.rejected_marker
    other = args.rejected_marker if result["accepted"] else args.accepted_marker
    other.unlink(missing_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        f"decision={'accepted' if result['accepted'] else 'rejected'}\n"
        f"result={args.output.resolve()}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

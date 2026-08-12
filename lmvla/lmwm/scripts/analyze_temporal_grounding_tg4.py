#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ARMS = (
    "clean_base",
    "future_off",
    "auxiliary_only",
    "conditioning_only",
    "parameter_matched_null",
    "full",
)
TRAIN_SEEDS = (1100, 1101, 1102)
EVAL_SEEDS = (0, 1, 2, 3)
COMPARISONS: tuple[tuple[str, dict[str, float]], ...] = (
    ("pretraining", {"future_off_normal": 1.0, "clean_base_normal": -1.0}),
    (
        "auxiliary_shaping",
        {"auxiliary_only_normal": 1.0, "parameter_matched_null_normal": -1.0},
    ),
    (
        "conditioning_without_auxiliary",
        {"conditioning_only_normal": 1.0, "parameter_matched_null_normal": -1.0},
    ),
    ("full_total", {"full_normal": 1.0, "parameter_matched_null_normal": -1.0}),
    ("full_vs_historical_off", {"full_normal": 1.0, "future_off_normal": -1.0}),
    (
        "route_interaction",
        {
            "full_normal": 1.0,
            "auxiliary_only_normal": -1.0,
            "conditioning_only_normal": -1.0,
            "parameter_matched_null_normal": 1.0,
        },
    ),
    ("content_use", {"full_normal": 1.0, "full_shuffled": -1.0}),
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260812


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_panel(root: Path, scene_manifest: dict[str, Any], scene_hash: str) -> dict:
    tasks = tuple(scene_manifest["tasks"])
    records: dict[tuple[int, str, int], float] = {}
    for path in sorted(root.glob("seed*/**/tasks/*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = str(payload["task_name"])
        fixed = payload.get("fixed_seed_manifest")
        if not isinstance(fixed, dict) or fixed.get("sha256") != scene_hash:
            raise ValueError(f"Missing or wrong fixed scene manifest in {path}")
        eval_seed = int(fixed["eval_seed"])
        if eval_seed not in EVAL_SEEDS or task not in tasks:
            raise ValueError(f"Unexpected task/eval seed in {path}")
        expected = [
            int(value)
            for value in scene_manifest["eval_seeds"][str(eval_seed)][task]
        ]
        episodes = payload.get("episodes", [])
        if len(episodes) != len(expected) or payload.get("n_episodes") != len(expected):
            raise ValueError(f"Incomplete episode report: {path}")
        observed = {int(row["seed"]): float(bool(row["success"])) for row in episodes}
        if set(observed) != set(expected):
            raise ValueError(f"Fixed scene mismatch in {path}")
        for scene_seed in expected:
            key = (eval_seed, task, scene_seed)
            if key in records:
                raise ValueError(f"Duplicate outcome key {key} in {root}")
            records[key] = observed[scene_seed]
    expected_count = len(EVAL_SEEDS) * len(tasks) * int(scene_manifest["episodes_per_cell"])
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} outcomes in {root}, got {len(records)}")
    return {"tasks": tasks, "records": records}


def contrast_records(
    panels: dict[str, dict[int, dict]], coefficients: dict[str, float]
) -> dict[tuple[int, int, str, int], float]:
    output: dict[tuple[int, int, str, int], float] = {}
    for train_seed in TRAIN_SEEDS:
        selected = [panels[name][train_seed]["records"] for name in coefficients]
        keys = set(selected[0])
        if any(set(records) != keys for records in selected[1:]):
            raise ValueError(f"Panel keys are not paired for training seed {train_seed}")
        for eval_seed, task, scene_seed in keys:
            output[(train_seed, eval_seed, task, scene_seed)] = sum(
                coefficients[name] * panels[name][train_seed]["records"][(eval_seed, task, scene_seed)]
                for name in coefficients
            )
    return output


def hierarchical_draws(
    records: dict[tuple[int, int, str, int], float],
    tasks: tuple[str, ...],
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    cells: dict[tuple[int, str, int], np.ndarray] = {}
    for train_seed in TRAIN_SEEDS:
        for task in tasks:
            for eval_seed in EVAL_SEEDS:
                values = [
                    value
                    for (observed_seed, observed_eval, observed_task, _), value in records.items()
                    if observed_seed == train_seed
                    and observed_eval == eval_seed
                    and observed_task == task
                ]
                if not values:
                    raise ValueError(f"Empty hierarchy cell {(train_seed, task, eval_seed)}")
                cells[(train_seed, task, eval_seed)] = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        values = []
        for seed_index in rng.integers(0, len(TRAIN_SEEDS), size=len(TRAIN_SEEDS)):
            train_seed = TRAIN_SEEDS[int(seed_index)]
            for task_index in rng.integers(0, len(tasks), size=len(tasks)):
                task = tasks[int(task_index)]
                for eval_index in rng.integers(0, len(EVAL_SEEDS), size=len(EVAL_SEEDS)):
                    eval_seed = EVAL_SEEDS[int(eval_index)]
                    episodes = cells[(train_seed, task, eval_seed)]
                    episode_indices = rng.integers(0, len(episodes), size=len(episodes))
                    values.append(float(episodes[episode_indices].mean()))
        draws[draw_index] = float(np.mean(values))
    return draws


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["paired_p"])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * rows[index]["paired_p"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def analyze(
    panels: dict[str, dict[int, dict]],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    expected_panels = {f"{arm}_normal" for arm in ARMS} | {"full_shuffled"}
    if set(panels) != expected_panels:
        raise ValueError(f"Panel set mismatch: {set(panels)} != {expected_panels}")
    tasks = tuple(panels["full_normal"][TRAIN_SEEDS[0]]["tasks"])
    rows = []
    for index, (name, coefficients) in enumerate(COMPARISONS):
        records = contrast_records(panels, coefficients)
        draws = hierarchical_draws(
            records,
            tasks,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        lower, upper = np.quantile(draws, [0.025, 0.975])
        nonpositive = (int(np.count_nonzero(draws <= 0.0)) + 1) / (len(draws) + 1)
        nonnegative = (int(np.count_nonzero(draws >= 0.0)) + 1) / (len(draws) + 1)
        cell_effects = {
            str(train_seed): {
                task: float(
                    np.mean(
                        [
                            value
                            for (seed_value, _, task_value, _), value in records.items()
                            if seed_value == train_seed and task_value == task
                        ]
                    )
                )
                for task in tasks
            }
            for train_seed in TRAIN_SEEDS
        }
        rows.append(
            {
                "name": name,
                "coefficients": coefficients,
                "mean_effect": float(np.mean(list(records.values()))),
                "hierarchical_ci95": [float(lower), float(upper)],
                "paired_p": min(1.0, 2.0 * min(nonpositive, nonnegative)),
                "training_seed_task_effects": cell_effects,
                "minimum_training_seed_task_effect": min(
                    value for by_task in cell_effects.values() for value in by_task.values()
                ),
                "task_safety_passed": all(
                    value >= -0.05
                    for by_task in cell_effects.values()
                    for value in by_task.values()
                ),
                "bootstrap": {
                    "samples": bootstrap_samples,
                    "random_seed": bootstrap_seed + index,
                    "levels": [
                        "training_seed",
                        "task_within_training_seed",
                        "eval_seed_within_task",
                        "paired_episode",
                    ],
                },
            }
        )
    holm_adjust(rows)
    for row in rows:
        row["accepted"] = (
            row["hierarchical_ci95"][0] > 0.0
            and row["holm_adjusted_p"] < 0.05
            and row["task_safety_passed"]
        )
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg4_source_decomposition_analysis_v1",
        "complete": True,
        "training_seeds": list(TRAIN_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "tasks": list(tasks),
        "holm_family": [name for name, _ in COMPARISONS],
        "comparisons": {row["name"]: row for row in rows},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    scene_path = repo / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    scene_bytes = scene_path.read_bytes()
    scene_manifest = json.loads(scene_bytes)
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    eval_root = repo / "lmvla/lawam/results/eval_runs/robotwin"
    panels: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        panels[f"{arm}_normal"] = {
            train_seed: load_panel(
                eval_root / f"temporal_grounding_tg4_{arm}_seed{train_seed}_normal",
                scene_manifest,
                scene_hash,
            )
            for train_seed in TRAIN_SEEDS
        }
    panels["full_shuffled"] = {
        train_seed: load_panel(
            eval_root / f"temporal_grounding_tg4_full_seed{train_seed}_shuffled",
            scene_manifest,
            scene_hash,
        )
        for train_seed in TRAIN_SEEDS
    }
    result = analyze(panels, bootstrap_samples=args.bootstrap_samples)
    atomic_json(args.output.resolve(), result)
    marker_lines = [
        "validated=true",
        f"protocol={result['protocol']}",
        *(
            f"{name}={str(row['accepted']).lower()}"
            for name, row in result["comparisons"].items()
        ),
    ]
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    args.marker.write_text("\n".join(marker_lines) + "\n", encoding="utf-8")
    print(json.dumps({name: row["accepted"] for name, row in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()

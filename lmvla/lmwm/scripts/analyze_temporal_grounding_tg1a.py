#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import atomic_write_text, exact_mcnemar, holm_adjust  # noqa: E402


CONDITIONS = ("normal", "shuffled", "null", "persistence")
CONTROLS = ("shuffled", "null", "persistence")
EVAL_SEEDS = (0, 1, 2, 3)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260807


def _summary_files(root: Path) -> list[Path]:
    return sorted(root.glob("seed*/**/tasks/*/summary.json"))


def load_condition(root: Path, scene_manifest: dict, scene_hash: str) -> dict:
    records: dict[tuple[int, str, int], bool] = {}
    tasks = tuple(scene_manifest["tasks"])
    for path in _summary_files(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = str(payload["task_name"])
        fixed = payload.get("fixed_seed_manifest")
        if not isinstance(fixed, dict) or fixed.get("sha256") != scene_hash:
            raise ValueError(f"Missing or wrong fixed scene manifest in {path}")
        eval_seed = int(fixed["eval_seed"])
        if eval_seed not in EVAL_SEEDS or task not in tasks:
            raise ValueError(f"Unexpected eval seed/task in {path}")
        expected_seeds = [int(value) for value in scene_manifest["eval_seeds"][str(eval_seed)][task]]
        episodes = payload.get("episodes", [])
        if len(episodes) != len(expected_seeds) or int(payload.get("n_episodes", -1)) != len(expected_seeds):
            raise ValueError(f"Incomplete episode report: {path}")
        by_seed = {int(row["seed"]): bool(row["success"]) for row in episodes}
        if set(by_seed) != set(expected_seeds):
            raise ValueError(f"Scene seed mismatch in {path}")
        for scene_seed in expected_seeds:
            key = (eval_seed, task, scene_seed)
            if key in records:
                raise ValueError(f"Duplicate outcome key {key}")
            records[key] = by_seed[scene_seed]
    expected_count = len(EVAL_SEEDS) * len(tasks) * int(scene_manifest["episodes_per_cell"])
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} outcomes under {root}, got {len(records)}")
    return {"records": records, "tasks": tasks}


def paired_hierarchy(normal: dict, control: dict) -> dict[str, dict[int, np.ndarray]]:
    if set(normal["records"]) != set(control["records"]):
        raise ValueError("Condition outcome keys are not exactly paired")
    paired: dict[str, dict[int, np.ndarray]] = {}
    for task in normal["tasks"]:
        paired[task] = {}
        for eval_seed in EVAL_SEEDS:
            keys = sorted(
                key for key in normal["records"] if key[0] == eval_seed and key[1] == task
            )
            paired[task][eval_seed] = np.asarray(
                [int(normal["records"][key]) - int(control["records"][key]) for key in keys],
                dtype=np.float64,
            )
    return paired


def hierarchical_ci(
    paired: dict[str, dict[int, np.ndarray]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tasks = tuple(sorted(paired))
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        task_indices = rng.integers(0, len(tasks), size=len(tasks))
        values = []
        for task_index in task_indices:
            task = tasks[int(task_index)]
            eval_indices = rng.integers(0, len(EVAL_SEEDS), size=len(EVAL_SEEDS))
            for eval_index in eval_indices:
                episodes = paired[task][EVAL_SEEDS[int(eval_index)]]
                episode_indices = rng.integers(0, len(episodes), size=len(episodes))
                values.append(float(episodes[episode_indices].mean()))
        draws[draw_index] = float(np.mean(values))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def analyze(
    reports: dict[str, dict],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    if tuple(reports) != CONDITIONS:
        raise ValueError(f"Conditions must be supplied in frozen order {CONDITIONS}")
    normal = reports["normal"]
    comparisons = {}
    tests = []
    for index, control_name in enumerate(CONTROLS):
        control = reports[control_name]
        paired = paired_hierarchy(normal, control)
        task_effects = {
            task: float(np.mean([values.mean() for values in eval_map.values()]))
            for task, eval_map in paired.items()
        }
        eval_seed_effects = {
            str(eval_seed): float(
                np.mean([paired[task][eval_seed].mean() for task in normal["tasks"]])
            )
            for eval_seed in EVAL_SEEDS
        }
        lower, upper = hierarchical_ci(
            paired,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        normal_map = {key: value for key, value in normal["records"].items()}
        control_map = {key: value for key, value in control["records"].items()}
        test = exact_mcnemar(normal_map, control_map)
        test["comparison"] = f"normal_minus_{control_name}"
        tests.append(test)
        comparisons[control_name] = {
            "normal_success_rate": float(np.mean(list(normal_map.values()))),
            "control_success_rate": float(np.mean(list(control_map.values()))),
            "mean_effect": float(np.mean(list(task_effects.values()))),
            "task_effects": task_effects,
            "eval_seed_effects": eval_seed_effects,
            "hierarchical_paired_bootstrap": {
                "levels": ["task", "eval_seed_within_task", "paired_episode"],
                "samples": bootstrap_samples,
                "random_seed": bootstrap_seed + index,
                "ci95": [lower, upper],
            },
        }
    holm_adjust(tests)
    tests_by_name = {row["comparison"].removeprefix("normal_minus_"): row for row in tests}
    for name, comparison in comparisons.items():
        comparison["pooled_exact_mcnemar"] = tests_by_name[name]
        comparison["accepted"] = (
            comparison["hierarchical_paired_bootstrap"]["ci95"][0] > 0.0
            and comparison["pooled_exact_mcnemar"]["holm_adjusted_p"] < 0.05
        )
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg1a_released_checkpoint_content_panel_v1",
        "complete": True,
        "checkpoint_selection": "frozen released LaWAM checkpoint; no training or checkpoint selection",
        "tasks": list(normal["tasks"]),
        "eval_seeds": list(EVAL_SEEDS),
        "episodes_per_condition": len(normal["records"]),
        "holm_family": [f"normal_minus_{name}" for name in CONTROLS],
        "comparisons": comparisons,
        "claim_gates": {
            "correct_future_content_used": comparisons["shuffled"]["accepted"],
            "future_route_necessary": comparisons["null"]["accepted"],
            "persistence_beaten": comparisons["persistence"]["accepted"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition}", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    scene_text = args.scene_manifest.read_bytes()
    import hashlib

    scene_hash = hashlib.sha256(scene_text).hexdigest()
    scene_manifest = json.loads(scene_text)
    reports = {
        condition: load_condition(getattr(args, condition), scene_manifest, scene_hash)
        for condition in CONDITIONS
    }
    result = analyze(reports, bootstrap_samples=args.bootstrap_samples)
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        args.marker,
        "validated=true\n"
        + "\n".join(
            f"{key}={str(value).lower()}" for key, value in result["claim_gates"].items()
        )
        + "\n",
    )
    print(json.dumps(result["claim_gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

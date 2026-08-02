#!/usr/bin/env python3
"""Summarize the matched pi0.5 RoboTwin confirmatory training-seed matrix."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


REPORT_PATTERNS = {
    "a0": re.compile(r"^pi05_rt_a0_public_exact_seed(\d+)\.json$"),
    "a2_abs": re.compile(r"^pi05_rt_a2_abs_confirmatory_s(\d+)\.json$"),
    "a3_live": re.compile(r"^pi05_rt_a3_live_confirmatory_s(\d+)\.json$"),
}
EXPECTED_TRAINING_SEEDS = (1000, 1001, 1002)
CONTRASTS = (("a2_abs", "a0"), ("a3_live", "a0"))


def load_reports(report_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    reports: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in sorted(report_dir.glob("*.json")):
        for method, pattern in REPORT_PATTERNS.items():
            match = pattern.match(path.name)
            if not match:
                continue
            report = json.loads(path.read_text())
            if report.get("summary_count") != 24 or report.get("task_count") != 6:
                break
            reports[method][int(match.group(1))] = report
            break
    return reports


def method_summary(seed_reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seed_rows = {}
    task_rates: dict[str, list[float]] = defaultdict(list)
    macros = []
    for seed, report in sorted(seed_reports.items()):
        macro = float(report["macro_success_rate"])
        macros.append(macro)
        rates = {
            task: float(row["mean_success_rate"])
            for task, row in sorted(report["tasks"].items())
        }
        for task, rate in rates.items():
            task_rates[task].append(rate)
        seed_rows[str(seed)] = {
            "macro_success_rate": macro,
            "micro_success_rate": float(report["micro_success_rate"]),
            "total_episodes": int(report["total_episodes"]),
            "task_success_rates": rates,
        }
    return {
        "training_seed_count": len(seed_rows),
        "mean_macro_success_rate": mean(macros) if macros else None,
        "macro_population_std": pstdev(macros) if len(macros) > 1 else None,
        "task_mean_success_rates": {
            task: mean(values) for task, values in sorted(task_rates.items())
        },
        "seeds": seed_rows,
    }


def outcome_map(report: dict[str, Any]) -> dict[str, dict[tuple[int, int], bool]]:
    by_task: dict[str, dict[tuple[int, int], bool]] = {}
    for task, task_row in report["tasks"].items():
        outcomes: dict[tuple[int, int], bool] = {}
        for cell in task_row["cells"]:
            eval_seed = cell.get("eval_seed")
            for episode in cell.get("episode_outcomes", []):
                scene_seed = episode.get("scene_seed")
                if eval_seed is None or scene_seed is None:
                    continue
                key = (int(eval_seed), int(scene_seed))
                if key in outcomes:
                    raise ValueError(f"duplicate episode key method task={task} key={key}")
                outcomes[key] = bool(episode["success"])
        by_task[task] = outcomes
    return by_task


def paired_hierarchical_contrast(
    candidate_reports: dict[int, dict[str, Any]],
    baseline_reports: dict[int, dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    common_seeds = sorted(set(candidate_reports) & set(baseline_reports))
    if not common_seeds:
        return {"available": False, "reason": "no common training seeds"}

    paired: dict[int, dict[str, np.ndarray]] = {}
    unmatched = 0
    for training_seed in common_seeds:
        candidate = outcome_map(candidate_reports[training_seed])
        baseline = outcome_map(baseline_reports[training_seed])
        tasks = sorted(set(candidate) & set(baseline))
        paired[training_seed] = {}
        for task in tasks:
            candidate_keys = set(candidate[task])
            baseline_keys = set(baseline[task])
            keys = sorted(candidate_keys & baseline_keys)
            unmatched += len(candidate_keys ^ baseline_keys)
            if keys:
                paired[training_seed][task] = np.asarray(
                    [int(candidate[task][key]) - int(baseline[task][key]) for key in keys],
                    dtype=np.float64,
                )

    if any(len(tasks) != 6 for tasks in paired.values()):
        return {
            "available": False,
            "reason": "episode outcomes are missing for one or more task/seed cells",
            "common_training_seeds": common_seeds,
        }

    per_seed = {
        str(seed): mean(float(values.mean()) for values in tasks.values())
        for seed, tasks in paired.items()
    }
    point = mean(per_seed.values())
    rng = np.random.default_rng(bootstrap_seed)
    training_seed_count = len(common_seeds)
    seed_bootstrap = []
    for seed in common_seeds:
        seed_draws = np.zeros((bootstrap_samples, training_seed_count))
        for values in paired[seed].values():
            indices = rng.integers(
                0,
                len(values),
                size=(bootstrap_samples, training_seed_count, len(values)),
            )
            seed_draws += values[indices].mean(axis=2) / len(paired[seed])
        seed_bootstrap.append(seed_draws)
    seed_bootstrap_array = np.stack(seed_bootstrap, axis=1)
    sampled_seed_indices = rng.integers(
        0,
        training_seed_count,
        size=(bootstrap_samples, training_seed_count),
    )
    rows = np.arange(bootstrap_samples)[:, None]
    positions = np.arange(training_seed_count)[None, :]
    draws = seed_bootstrap_array[rows, sampled_seed_indices, positions].mean(axis=1)
    paired_episodes = sum(len(values) for tasks in paired.values() for values in tasks.values())
    return {
        "available": True,
        "common_training_seeds": common_seeds,
        "paired_episode_count": paired_episodes,
        "unmatched_episode_key_count": unmatched,
        "point_estimate_macro_delta": point,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "per_training_seed_macro_delta": per_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "hierarchy": "resample training seeds, then paired episodes within each task",
    }


def summarize(
    report_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    reports = load_reports(report_dir)
    missing = [
        f"{method}:seed{seed}"
        for method in REPORT_PATTERNS
        for seed in EXPECTED_TRAINING_SEEDS
        if seed not in reports.get(method, {})
    ]
    methods = {
        method: method_summary(reports.get(method, {}))
        for method in REPORT_PATTERNS
    }
    contrasts = {}
    for index, (candidate, baseline) in enumerate(CONTRASTS):
        contrasts[f"{candidate}_minus_{baseline}"] = paired_hierarchical_contrast(
            reports.get(candidate, {}),
            reports.get(baseline, {}),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
    return {
        "complete": not missing,
        "expected_training_seeds": list(EXPECTED_TRAINING_SEEDS),
        "missing_reports": missing,
        "methods": methods,
        "contrasts": contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()
    report = summarize(
        args.report_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

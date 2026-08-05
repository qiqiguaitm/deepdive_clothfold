#!/usr/bin/env python3
"""Analyze the frozen multistage-versus-control MT6 scope interaction."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


EXPECTED_SEEDS = (1000, 1001, 1002)


def load_matrix_module():
    path = Path(__file__).with_name("summarize_pi05_confirmatory_matrix.py")
    spec = importlib.util.spec_from_file_location("pi05_confirmatory_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate training seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, got {sorted(result)}")
    return result


def validate_scope(scope: dict[str, Any], manifest: dict[str, Any]) -> dict[str, list[str]]:
    groups = {name: list(tasks) for name, tasks in scope["groups"].items()}
    if set(groups) != {"multistage_aliasing", "reactive_geometric_control"}:
        raise ValueError("scope must define the two frozen task groups")
    flattened = [task for tasks in groups.values() for task in tasks]
    if len(flattened) != len(set(flattened)):
        raise ValueError("scope task groups overlap")
    if set(flattened) != set(manifest["tasks"]):
        raise ValueError("scope task groups must partition the frozen manifest")
    if any(len(tasks) != 3 for tasks in groups.values()):
        raise ValueError("each frozen scope group must contain three tasks")
    return groups


def analyze(
    candidate: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    matrix = load_matrix_module()
    groups = validate_scope(scope, manifest)
    audits = {}
    for method, reports in (("candidate", candidate), ("baseline", baseline)):
        if set(reports) != set(EXPECTED_SEEDS):
            raise ValueError(f"{method} reports do not cover frozen training seeds")
        for seed, report in sorted(reports.items()):
            audit = matrix.audit_report(report, manifest)
            audits[f"{method}:seed{seed}"] = audit
            if not audit["accepted"]:
                raise ValueError(f"protocol audit rejected {method}:seed{seed}: {audit['errors']}")

    paired: dict[int, dict[str, np.ndarray]] = {}
    for seed in EXPECTED_SEEDS:
        candidate_outcomes = matrix.outcome_map(candidate[seed])
        baseline_outcomes = matrix.outcome_map(baseline[seed])
        paired[seed] = {}
        for task in manifest["tasks"]:
            if set(candidate_outcomes[task]) != set(baseline_outcomes[task]):
                raise ValueError(f"unmatched episode keys seed={seed} task={task}")
            keys = sorted(candidate_outcomes[task])
            paired[seed][task] = np.asarray(
                [
                    int(candidate_outcomes[task][key])
                    - int(baseline_outcomes[task][key])
                    for key in keys
                ],
                dtype=np.float64,
            )

    per_seed = {
        str(seed): {
            group: mean(float(paired[seed][task].mean()) for task in tasks)
            for group, tasks in groups.items()
        }
        for seed in EXPECTED_SEEDS
    }
    for row in per_seed.values():
        row["scope_interaction"] = (
            row["multistage_aliasing"] - row["reactive_geometric_control"]
        )

    samples = int(scope["statistics"]["bootstrap_samples"])
    bootstrap_seed = int(scope["statistics"]["bootstrap_seed"])
    rng = np.random.default_rng(bootstrap_seed)
    seed_count = len(EXPECTED_SEEDS)
    source_draws: dict[str, list[np.ndarray]] = {group: [] for group in groups}
    for seed in EXPECTED_SEEDS:
        for group, tasks in groups.items():
            draws = np.zeros((samples, seed_count), dtype=np.float64)
            for task in tasks:
                values = paired[seed][task]
                indices = rng.integers(
                    0, len(values), size=(samples, seed_count, len(values))
                )
                draws += values[indices].mean(axis=2) / len(tasks)
            source_draws[group].append(draws)

    sampled_seed_indices = rng.integers(
        0, seed_count, size=(samples, seed_count)
    )
    rows = np.arange(samples)[:, None]
    positions = np.arange(seed_count)[None, :]
    bootstrap = {}
    for group in groups:
        stacked = np.stack(source_draws[group], axis=1)
        bootstrap[group] = stacked[rows, sampled_seed_indices, positions].mean(axis=1)
    bootstrap["scope_interaction"] = (
        bootstrap["multistage_aliasing"]
        - bootstrap["reactive_geometric_control"]
    )

    point = {
        group: mean(float(per_seed[str(seed)][group]) for seed in EXPECTED_SEEDS)
        for group in groups
    }
    point["scope_interaction"] = (
        point["multistage_aliasing"] - point["reactive_geometric_control"]
    )
    intervals = {
        key: [float(value) for value in np.percentile(draws, [2.5, 97.5])]
        for key, draws in bootstrap.items()
    }
    return {
        "complete": True,
        "scope_version": scope["version"],
        "groups": groups,
        "protocol_audits": audits,
        "paired_episodes": sum(
            len(values) for tasks in paired.values() for values in tasks.values()
        ),
        "per_training_seed_delta": per_seed,
        "point_estimate": point,
        "ci95": intervals,
        "scope_interaction_positive": point["scope_interaction"] > 0.0,
        "scope_interaction_ci95_excludes_zero": intervals["scope_interaction"][0] > 0.0,
        "interpretation": scope["interpretation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--baseline", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_paths = parse_seed_paths(args.candidate)
    baseline_paths = parse_seed_paths(args.baseline)
    result = analyze(
        {seed: json.loads(path.read_text()) for seed, path in candidate_paths.items()},
        {seed: json.loads(path.read_text()) for seed, path in baseline_paths.items()},
        json.loads(args.manifest.read_text()),
        json.loads(args.scope.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

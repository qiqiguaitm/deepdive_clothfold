#!/usr/bin/env python3
"""Audit and gate the three-seed MT1 oracle-transition comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (1000, 1001, 1002)
PREDECLARED_MULTISTAGE_TASKS = (
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
)


def load_matrix_module():
    path = Path(__file__).with_name("summarize_pi05_confirmatory_matrix.py")
    spec = importlib.util.spec_from_file_location("pi05_confirmatory_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate training seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, got {sorted(result)}")
    return result


def gate_decision(contrast: dict[str, Any], task_deltas: dict[str, float]) -> dict[str, Any]:
    ci = contrast.get("ci95") if contrast.get("available") else None
    improved = sorted(task for task, delta in task_deltas.items() if delta > 0.0)
    checks = {
        "three_seed_contrast_available": bool(contrast.get("available")),
        "point_estimate_positive": bool(contrast.get("available"))
        and float(contrast["point_estimate_macro_delta"]) > 0.0,
        "ci95_excludes_zero": ci is not None and float(ci[0]) > 0.0,
        "at_least_two_multistage_tasks_improve": len(improved) >= 2,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "improved_multistage_tasks": improved,
    }


def analyze(
    candidate: dict[int, dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    matrix = load_matrix_module()
    audits = {}
    for method, reports in (("candidate", candidate), ("baseline", baseline)):
        for seed, report in sorted(reports.items()):
            audits[f"{method}:seed{seed}"] = matrix.audit_report(report, manifest)
    rejected = [name for name, audit in audits.items() if not audit["accepted"]]
    if rejected:
        raise ValueError(f"protocol audit rejected reports: {rejected}")

    contrast = matrix.paired_hierarchical_contrast(
        candidate,
        baseline,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    task_deltas = {}
    for task in PREDECLARED_MULTISTAGE_TASKS:
        candidate_mean = sum(float(candidate[seed]["tasks"][task]["mean_success_rate"]) for seed in EXPECTED_SEEDS) / 3
        baseline_mean = sum(float(baseline[seed]["tasks"][task]["mean_success_rate"]) for seed in EXPECTED_SEEDS) / 3
        task_deltas[task] = candidate_mean - baseline_mean
    decision = gate_decision(contrast, task_deltas)
    return {
        "complete": True,
        "protocol_audits": audits,
        "contrast": contrast,
        "predeclared_multistage_mean_delta": task_deltas,
        "gate": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--baseline", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    args = parser.parse_args()
    if not args.pilot_gate.is_file():
        raise FileNotFoundError(args.pilot_gate)
    candidate_paths = parse_seed_paths(args.candidate)
    baseline_paths = parse_seed_paths(args.baseline)
    candidate = {seed: json.loads(path.read_text()) for seed, path in candidate_paths.items()}
    baseline = {seed: json.loads(path.read_text()) for seed, path in baseline_paths.items()}
    result = analyze(
        candidate,
        baseline,
        json.loads(args.manifest.read_text()),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["pilot_gate"] = str(args.pilot_gate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.accepted_marker.unlink(missing_ok=True)
    if result["gate"]["accepted"]:
        args.accepted_marker.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_marker.write_text(f"accepted=true\nresult={args.output.resolve()}\n")


if __name__ == "__main__":
    main()

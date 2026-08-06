#!/usr/bin/env python3
"""Audit and compare P2 candidates with the frozen public pi0.5 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mt_transition_controls import atomic_write_text, outcome_map  # noqa: E402
from analyze_pi05_predictive_adapter_p2 import validate_report  # noqa: E402


TRAIN_SEEDS = (1000, 1001, 1002)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260806


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_delta(candidate: dict, baseline: dict, task: str) -> np.ndarray:
    candidate_map = outcome_map(candidate, task)
    baseline_map = outcome_map(baseline, task)
    if set(candidate_map) != set(baseline_map):
        raise ValueError(f"candidate and public scene keys differ for task {task}")
    keys = sorted(candidate_map)
    return np.asarray(
        [candidate_map[key] - baseline_map[key] for key in keys], dtype=np.float64
    )


def analyze(
    public: dict,
    candidates: dict[int, dict],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    validate_report("public_pi05", public)
    if tuple(sorted(candidates)) != TRAIN_SEEDS:
        raise ValueError(f"candidate seeds must be exactly {TRAIN_SEEDS}")
    public_keys = set(outcome_map(public))
    for seed, report in candidates.items():
        validate_report(f"candidate_seed{seed}", report)
        if set(outcome_map(report)) != public_keys:
            raise ValueError(f"candidate seed {seed} does not exactly match public keys")

    tasks = sorted(public["tasks"])
    seed_effects = {
        str(seed): candidates[seed]["macro_success_rate"]
        - public["macro_success_rate"]
        for seed in TRAIN_SEEDS
    }
    task_effects = {
        str(seed): {
            task: candidates[seed]["tasks"][task]["mean_success_rate"]
            - public["tasks"][task]["mean_success_rate"]
            for task in tasks
        }
        for seed in TRAIN_SEEDS
    }

    rng = np.random.default_rng(bootstrap_seed)
    per_seed = np.zeros((len(TRAIN_SEEDS), bootstrap_samples), dtype=np.float64)
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        for task in tasks:
            delta = _paired_delta(candidates[seed], public, task)
            indices = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
            per_seed[seed_index] += delta[indices].mean(axis=1) / len(tasks)
    selected = rng.integers(
        0, len(TRAIN_SEEDS), size=(bootstrap_samples, len(TRAIN_SEEDS))
    )
    rows = np.arange(bootstrap_samples)[:, None]
    hierarchical = per_seed.T[rows, selected].mean(axis=1)
    lower, upper = np.quantile(hierarchical, [0.025, 0.975])
    accepted = float(lower) > 0.0
    return {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_p5_public_reference_v1",
        "complete": True,
        "public_macro_success_rate": public["macro_success_rate"],
        "candidate_macro_success_rates": {
            str(seed): candidates[seed]["macro_success_rate"] for seed in TRAIN_SEEDS
        },
        "seed_effects_vs_public": seed_effects,
        "task_effects_vs_public": task_effects,
        "mean_seed_effect_vs_public": float(np.mean(list(seed_effects.values()))),
        "hierarchical_paired_bootstrap": {
            "unit_levels": ["candidate_training_seed", "paired_episode_within_task"],
            "samples": bootstrap_samples,
            "random_seed": bootstrap_seed,
            "ci95": [float(lower), float(upper)],
        },
        "checks": {
            "public_has_24_cells_and_1200_episodes": True,
            "all_candidate_seeds_present": True,
            "exact_scene_pairing": True,
            "hierarchical_95ci_excludes_zero_positive": accepted,
        },
        "accepted": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.public.read_text())
    candidates: dict[int, dict] = {}
    candidate_paths: dict[str, str] = {}
    for value in args.candidate:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        path = Path(path_text)
        if seed in candidates:
            raise ValueError(f"duplicate candidate seed: {seed}")
        candidates[seed] = json.loads(path.read_text())
        candidate_paths[str(seed)] = str(path)

    result = analyze(public, candidates, bootstrap_samples=args.bootstrap_samples)
    audit = {
        "schema_version": 1,
        "protocol": "pi05_predictive_adapter_p5_public_episode_audit_v1",
        "passed": result["checks"]["exact_scene_pairing"],
        "public_report": str(args.public),
        "public_report_sha256": sha256(args.public),
        "scene_manifest": str(args.scene_manifest),
        "scene_manifest_sha256": sha256(args.scene_manifest),
        "candidate_reports": candidate_paths,
        "summary_count": public["summary_count"],
        "total_episodes": public["total_episodes"],
        "task_count": len(public["tasks"]),
        "exact_episode_keys_vs_all_candidates": True,
        "rerun_required": False,
    }
    atomic_write_text(args.audit_output, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.marker,
        f"validated=true\naccepted={str(result['accepted']).lower()}\nreport={args.output}\n",
    )
    print(json.dumps({"audit": audit, "accepted": result["accepted"]}, indent=2))


if __name__ == "__main__":
    main()

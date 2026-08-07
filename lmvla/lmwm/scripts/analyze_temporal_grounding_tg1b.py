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
from analyze_temporal_grounding_tg1a import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    EVAL_SEEDS,
    hierarchical_ci,
    load_condition,
)


BOOTSTRAP_SEED = 20260808


def analyze(
    reports: dict[str, dict],
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict:
    required = ("future_off_e36", "future_off_e50", "local_wm_e36", "local_wm_e50")
    if tuple(reports) != required:
        raise ValueError(f"TG1B reports must be supplied in frozen order {required}")
    keys = set(reports[required[0]]["records"])
    if any(set(reports[name]["records"]) != keys for name in required[1:]):
        raise ValueError("TG1B panel is not exactly paired")
    tasks = reports[required[0]]["tasks"]
    paired: dict[str, dict[int, np.ndarray]] = {}
    for task in tasks:
        paired[task] = {}
        for eval_seed in EVAL_SEEDS:
            cell_keys = sorted(key for key in keys if key[0] == eval_seed and key[1] == task)
            paired[task][eval_seed] = np.asarray(
                [
                    (int(reports["local_wm_e50"]["records"][key]) - int(reports["local_wm_e36"]["records"][key]))
                    - (int(reports["future_off_e50"]["records"][key]) - int(reports["future_off_e36"]["records"][key]))
                    for key in cell_keys
                ],
                dtype=np.float64,
            )
    task_effects = {
        task: float(np.mean([values.mean() for values in eval_map.values()]))
        for task, eval_map in paired.items()
    }
    eval_seed_effects = {
        str(eval_seed): float(np.mean([paired[task][eval_seed].mean() for task in tasks]))
        for eval_seed in EVAL_SEEDS
    }
    lower, upper = hierarchical_ci(
        paired, samples=bootstrap_samples, seed=bootstrap_seed
    )
    rates = {
        name: float(np.mean(list(report["records"].values()))) for name, report in reports.items()
    }
    did_mean = float(np.mean(list(task_effects.values())))
    return {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg1b_execution_cadence_panel_v1",
        "complete": True,
        "training_seed": 2027,
        "tasks": list(tasks),
        "eval_seeds": list(EVAL_SEEDS),
        "episodes_per_panel_cell": len(keys),
        "success_rates": rates,
        "difference_in_differences": {
            "estimand": "(local_wm_e50-local_wm_e36)-(future_off_e50-future_off_e36)",
            "mean": did_mean,
            "task_effects": task_effects,
            "eval_seed_effects": eval_seed_effects,
            "hierarchical_paired_bootstrap": {
                "levels": ["task", "eval_seed_within_task", "paired_episode"],
                "samples": bootstrap_samples,
                "random_seed": bootstrap_seed,
                "ci95": [lower, upper],
            },
        },
        "claim_gate": {
            "local_wm_specific_cadence_sensitivity": lower > 0.0,
            "interpretation_boundary": "diagnostic only; does not establish future-content use",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("future-off-e36", "future-off-e50", "local-wm-e36", "local-wm-e50"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    args = parser.parse_args()
    scene_bytes = args.scene_manifest.read_bytes()
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    scene_manifest = json.loads(scene_bytes)
    paths = {
        "future_off_e36": args.future_off_e36,
        "future_off_e50": args.future_off_e50,
        "local_wm_e36": args.local_wm_e36,
        "local_wm_e50": args.local_wm_e50,
    }
    result = analyze(
        {
            name: load_condition(path, scene_manifest, scene_hash)
            for name, path in paths.items()
        },
        bootstrap_samples=args.bootstrap_samples,
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    accepted = result["claim_gate"]["local_wm_specific_cadence_sensitivity"]
    atomic_write_text(args.marker, f"validated=true\naccepted={str(accepted).lower()}\n")
    print(json.dumps(result["claim_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

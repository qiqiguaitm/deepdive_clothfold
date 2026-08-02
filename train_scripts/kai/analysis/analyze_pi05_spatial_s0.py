#!/usr/bin/env python3
"""Create the frozen paired verdict for the pi0.5 spatial S0 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("no_goal", "current", "privileged")
METRICS = ("endpoint_l2", "flow_loss", "action_cosine")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["task"]),
        int(row["episode_index"]),
        int(row["frame_index"]),
        int(row["target_frame_index"]),
    )


def episode_means(rows: list[dict], task: str, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        if row["task"] == task:
            grouped.setdefault(int(row["episode_index"]), []).append(float(row[metric]))
    return {episode: float(np.mean(values)) for episode, values in grouped.items()}


def paired_contrast(
    privileged: list[dict],
    control: list[dict],
    *,
    task: str,
    metric: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict:
    privileged_by_episode = episode_means(privileged, task, metric)
    control_by_episode = episode_means(control, task, metric)
    episodes = sorted(privileged_by_episode)
    if episodes != sorted(control_by_episode):
        raise ValueError(f"episode mismatch for {task}/{metric}")
    differences = np.asarray(
        [privileged_by_episode[index] - control_by_episode[index] for index in episodes],
        dtype=np.float64,
    )
    draws = rng.choice(differences, size=(bootstrap_samples, len(differences)), replace=True).mean(axis=1)
    return {
        "definition": "privileged_minus_control",
        "episode_count": len(episodes),
        "mean": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_probability_lt_zero": float(np.mean(draws < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("logs/spatial_s0"))
    parser.add_argument("--output", type=Path, default=Path("logs/spatial_s0/s0_offline_verdict.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()

    paths = {arm: args.input_dir / f"eval_{arm}_full.json" for arm in ARMS}
    results = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    reference = results["no_goal"]
    reference_keys = [sample_key(row) for row in reference["samples"]]
    for arm, result in results.items():
        if not result["complete"] or int(result["sample_count"]) != 320:
            raise ValueError(f"incomplete result for {arm}")
        if result["manifest_sha256"] != reference["manifest_sha256"]:
            raise ValueError(f"manifest mismatch for {arm}")
        if result["evaluator_sha256"] != reference["evaluator_sha256"]:
            raise ValueError(f"evaluator mismatch for {arm}")
        if [sample_key(row) for row in result["samples"]] != reference_keys:
            raise ValueError(f"sample ordering mismatch for {arm}")

    rng = np.random.default_rng(args.bootstrap_seed)
    contrasts = {}
    for task in ("hammer", "stack_three"):
        contrasts[task] = {}
        for control in ("no_goal", "current"):
            contrasts[task][control] = {
                metric: paired_contrast(
                    results["privileged"]["samples"],
                    results[control]["samples"],
                    task=task,
                    metric=metric,
                    rng=rng,
                    bootstrap_samples=args.bootstrap_samples,
                )
                for metric in METRICS
            }

    stack_endpoint = {
        arm: float(results[arm]["by_task"]["stack_three"]["endpoint_l2"])
        for arm in ARMS
    }
    stack_requirement = stack_endpoint["privileged"] < min(
        stack_endpoint["no_goal"], stack_endpoint["current"]
    )
    verdict = {
        "schema_version": 1,
        "protocol": {
            "bootstrap_unit": "episode_mean_over_four_frozen_frames",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "manifest_sha256": reference["manifest_sha256"],
            "evaluator_sha256": reference["evaluator_sha256"],
            "input_sha256": {arm: sha256(path) for arm, path in paths.items()},
        },
        "arm_task_aggregates": {arm: results[arm]["by_task"] for arm in ARMS},
        "paired_contrasts": contrasts,
        "decision": {
            "stack_three_endpoint_privileged_better_than_both_controls": stack_requirement,
            "hammer_endpoint_privileged_worse_than_no_goal": (
                results["privileged"]["by_task"]["hammer"]["endpoint_l2"]
                > results["no_goal"]["by_task"]["hammer"]["endpoint_l2"]
            ),
            "hammer_endpoint_privileged_worse_than_current": (
                results["privileged"]["by_task"]["hammer"]["endpoint_l2"]
                > results["current"]["by_task"]["hammer"]["endpoint_l2"]
            ),
            "t3a_gate_pass": False if not stack_requirement else None,
            "next_action": (
                "stop_t3b_and_spatial_predictor_expansion"
                if not stack_requirement
                else "requires_preregistered_hammer_materiality_decision"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

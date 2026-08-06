from __future__ import annotations

import copy

from analyze_pi05_predictive_adapter_p5 import analyze


TASKS = [f"task_{index}" for index in range(6)]


def report(successes: int) -> dict:
    tasks = {}
    for task in TASKS:
        cells = []
        for eval_seed in range(4):
            outcomes = [
                {
                    "episode_id": episode_id,
                    "scene_seed": eval_seed * 100 + episode_id,
                    "success": episode_id < successes,
                }
                for episode_id in range(50)
            ]
            cells.append({"eval_seed": eval_seed, "episode_outcomes": outcomes})
        tasks[task] = {"cells": cells, "mean_success_rate": successes / 50}
    return {
        "summary_count": 24,
        "total_episodes": 1200,
        "macro_success_rate": successes / 50,
        "tasks": tasks,
    }


def test_accepts_positive_public_comparison() -> None:
    result = analyze(
        report(20),
        {seed: report(35) for seed in (1000, 1001, 1002)},
        bootstrap_samples=500,
    )
    assert result["accepted"]


def test_rejects_scene_mismatch() -> None:
    public = report(20)
    candidates = {seed: report(35) for seed in (1000, 1001, 1002)}
    broken = copy.deepcopy(candidates[1002])
    broken["tasks"][TASKS[0]]["cells"][0]["episode_outcomes"][0]["scene_seed"] = 999
    candidates[1002] = broken
    try:
        analyze(public, candidates, bootstrap_samples=10)
    except ValueError as error:
        assert "keys" in str(error)
    else:
        raise AssertionError("mismatched report passed")

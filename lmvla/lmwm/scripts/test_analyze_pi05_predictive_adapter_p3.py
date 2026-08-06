from __future__ import annotations

import copy

from analyze_pi05_predictive_adapter_p3 import analyze


TASKS = [f"task_{index}" for index in range(6)]


def make_report(successes: int) -> dict:
    tasks = {}
    total = 0
    for task in TASKS:
        cells = []
        for eval_seed in range(4):
            outcomes = []
            for episode_id in range(50):
                success = episode_id < successes
                total += int(success)
                outcomes.append(
                    {"episode_id": episode_id, "scene_seed": eval_seed * 1000 + episode_id, "success": success}
                )
            cells.append({"eval_seed": eval_seed, "episode_outcomes": outcomes})
        tasks[task] = {"cells": cells, "mean_success_rate": successes / 50}
    return {"summary_count": 24, "total_episodes": 1200, "macro_success_rate": total / 1200, "tasks": tasks}


def test_accepts_matched_positive_effect_and_task_safety() -> None:
    baselines = {seed: make_report(20 + index) for index, seed in enumerate((1000, 1001, 1002))}
    candidates = {seed: make_report(32 + index) for index, seed in enumerate((1000, 1001, 1002))}
    result = analyze(baselines, candidates, bootstrap_samples=500, bootstrap_seed=7)
    assert result["accepted"]
    assert result["task_safe"]
    assert result["hierarchical_paired_bootstrap"]["ci95"][0] > 0


def test_rejects_zero_effect() -> None:
    reports = {seed: make_report(25) for seed in (1000, 1001, 1002)}
    assert not analyze(reports, reports, bootstrap_samples=100, bootstrap_seed=7)["accepted"]


def test_rejects_episode_key_mismatch() -> None:
    baselines = {seed: make_report(20) for seed in (1000, 1001, 1002)}
    candidates = {seed: make_report(30) for seed in (1000, 1001, 1002)}
    broken = copy.deepcopy(candidates[1002])
    broken["tasks"][TASKS[0]]["cells"][0]["episode_outcomes"][0]["scene_seed"] = 999999
    candidates[1002] = broken
    try:
        analyze(baselines, candidates, bootstrap_samples=10)
    except ValueError as error:
        assert "scene keys" in str(error)
    else:
        raise AssertionError("mismatched episode keys were accepted")

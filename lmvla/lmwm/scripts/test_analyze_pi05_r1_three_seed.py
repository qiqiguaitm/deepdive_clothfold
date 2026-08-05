from __future__ import annotations

import copy

import pytest

from analyze_pi05_r1_three_seed import analyze


TASKS = [f"task_{index}" for index in range(6)]


def report(successes: int) -> dict:
    tasks = {}
    total = 0
    for task in TASKS:
        cells = []
        for eval_seed in range(4):
            rows = []
            for episode_id in range(50):
                success = episode_id < successes
                total += int(success)
                rows.append(
                    {
                        "scene_seed": eval_seed * 1000 + episode_id,
                        "success": success,
                    }
                )
            cells.append({"eval_seed": eval_seed, "episode_outcomes": rows})
        tasks[task] = {"cells": cells}
    return {"summary_count": 24, "total_episodes": 1200, "tasks": tasks}


def reports(combined_successes: int) -> dict:
    return {
        seed: {
            "a0": report(20),
            "predictive": report(22),
            "crave": report(24),
            "combined": report(combined_successes),
        }
        for seed in (1000, 1001, 1002)
    }


def test_accepts_consistent_combined_gain() -> None:
    result = analyze(reports(35), bootstrap_samples=200)
    assert result["accepted"]
    assert all(comparison["hierarchical_paired_bootstrap_ci95"][0] > 0 for comparison in result["comparisons"].values())


def test_rejects_combined_parity() -> None:
    result = analyze(reports(24), bootstrap_samples=100)
    assert not result["accepted"]


def test_rejects_scene_mismatch() -> None:
    payload = reports(35)
    broken = copy.deepcopy(payload[1001]["combined"])
    broken["tasks"][TASKS[0]]["cells"][0]["episode_outcomes"][0]["scene_seed"] = 999999
    payload[1001]["combined"] = broken
    with pytest.raises(ValueError, match="scene identities"):
        analyze(payload, bootstrap_samples=10)

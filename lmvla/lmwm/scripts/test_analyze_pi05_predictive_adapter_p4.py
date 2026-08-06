from __future__ import annotations

import copy

from analyze_pi05_predictive_adapter_p4 import CONTROLS, analyze


TASKS = [f"task_{index}" for index in range(6)]
SEEDS = (1000, 1001, 1002)


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
                outcomes.append({"scene_seed": eval_seed * 1000 + episode_id, "success": success})
            cells.append({"eval_seed": eval_seed, "episode_outcomes": outcomes})
        tasks[task] = {"cells": cells, "mean_success_rate": successes / 50}
    return {"summary_count": 24, "total_episodes": 1200, "macro_success_rate": total / 1200, "tasks": tasks}


def test_all_three_claim_gates_pass_for_strong_paired_effects() -> None:
    normal = {seed: make_report(40) for seed in SEEDS}
    controls = {condition: {seed: make_report(20) for seed in SEEDS} for condition in CONTROLS}
    result = analyze(normal, controls, bootstrap_samples=300, bootstrap_seed=7)
    assert all(result["claim_gates"].values())
    assert all(row["pooled_exact_mcnemar"]["holm_adjusted_p"] < 0.05 for row in result["comparisons"].values())


def test_route_can_pass_while_content_fails() -> None:
    normal = {seed: make_report(30) for seed in SEEDS}
    controls = {
        "shuffled": {seed: make_report(30) for seed in SEEDS},
        "zero_gate": {seed: make_report(15) for seed in SEEDS},
        "masked": {seed: make_report(30) for seed in SEEDS},
    }
    gates = analyze(normal, controls, bootstrap_samples=200, bootstrap_seed=7)["claim_gates"]
    assert not gates["content_specific_causality"]
    assert gates["route_necessity"]
    assert not gates["action_conditioning_use"]


def test_rejects_scene_mismatch() -> None:
    normal = {seed: make_report(30) for seed in SEEDS}
    controls = {condition: {seed: make_report(20) for seed in SEEDS} for condition in CONTROLS}
    broken = copy.deepcopy(controls["masked"][1001])
    broken["tasks"][TASKS[0]]["cells"][0]["episode_outcomes"][0]["scene_seed"] = 999999
    controls["masked"][1001] = broken
    try:
        analyze(normal, controls, bootstrap_samples=10)
    except ValueError as error:
        assert "scene keys" in str(error)
    else:
        raise AssertionError("mismatched episode keys were accepted")

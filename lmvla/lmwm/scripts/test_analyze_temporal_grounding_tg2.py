from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("analyze_temporal_grounding_tg2.py")
SPEC = importlib.util.spec_from_file_location("analyze_temporal_grounding_tg2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TASKS = tuple(f"task_{index}" for index in range(6))


def report(fn) -> dict:
    return {
        "tasks": TASKS,
        "records": {
            (eval_seed, task, scene): bool(fn(eval_seed, task_index, scene))
            for eval_seed in MODULE.EVAL_SEEDS
            for task_index, task in enumerate(TASKS)
            for scene in range(50)
        },
    }


def test_tg2_accepts_fixed_utility_and_horizon_effect() -> None:
    reports = {arm: {} for arm in MODULE.ARMS}
    for seed in MODULE.TRAIN_SEEDS:
        reports["future_off"][seed] = report(lambda *_: False)
        reports["fixed_endpoint"][seed] = report(lambda *_: True)
        reports["raw_milestone"][seed] = report(lambda _e, _t, scene: scene % 2 == 0)
    result = MODULE.analyze(reports, bootstrap_samples=120, bootstrap_seed=4)
    assert result["claim_gates"]["fixed_endpoint_utility"] is True
    assert result["claim_gates"]["target_horizon_effect"] is True
    assert result["stop_decision"]["tg3_authorized"] is True
    assert result["stop_decision"]["both_active_targets_fail_utility_gate"] is False


def test_tg2_task_safety_blocks_positive_macro() -> None:
    reports = {arm: {} for arm in MODULE.ARMS}
    for seed in MODULE.TRAIN_SEEDS:
        reports["future_off"][seed] = report(lambda *_: False)
        reports["raw_milestone"][seed] = report(lambda *_: False)
        reports["fixed_endpoint"][seed] = report(
            lambda _e, task, scene: task != 5 and scene < 40
        )
    # Make the baseline win on one task to force an unsafe seed/task effect.
    for seed in MODULE.TRAIN_SEEDS:
        for key in list(reports["future_off"][seed]["records"]):
            if key[1] == "task_5":
                reports["future_off"][seed]["records"][key] = True
    result = MODULE.analyze(reports, bootstrap_samples=80, bootstrap_seed=5)
    comparison = result["comparisons"]["fixed_endpoint_minus_future_off"]
    assert comparison["equal_training_seed_mean"] > 0.0
    assert comparison["task_safe"] is False
    assert comparison["accepted"] is False

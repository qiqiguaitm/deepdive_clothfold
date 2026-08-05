from __future__ import annotations

from copy import deepcopy

from analyze_pi05_predictive_adapter_p1 import apply_gate


TASKS = [f"task_{index}" for index in range(6)]


def _report(successes_per_task: dict[str, int]) -> dict:
    tasks = {}
    total_successes = 0
    for task_index, task in enumerate(TASKS):
        successes = successes_per_task[task]
        cells = []
        for seed in range(4):
            outcomes = []
            for episode in range(50):
                success = episode < successes
                total_successes += int(success)
                outcomes.append(
                    {
                        "episode_id": episode,
                        "scene_seed": task_index * 100_000 + seed * 1_000 + episode,
                        "success": success,
                    }
                )
            cells.append(
                {
                    "eval_seed": seed,
                    "episodes": 50,
                    "successes": successes,
                    "success_rate": successes / 50,
                    "episode_outcomes": outcomes,
                }
            )
        tasks[task] = {
            "cells": cells,
            "mean_success_rate": successes / 50,
            "total_episodes": 200,
            "total_successes": successes * 4,
        }
    macro = sum(successes_per_task.values()) / (len(TASKS) * 50)
    return {
        "macro_success_rate": macro,
        "micro_success_rate": macro,
        "summary_count": 24,
        "task_count": 6,
        "tasks": tasks,
        "total_episodes": 1200,
        "total_successes": total_successes,
    }


def test_p1_gate_accepts_only_when_all_predeclared_checks_pass() -> None:
    normal = _report({task: 40 for task in TASKS})
    controls = {
        "a0": _report({task: 39 for task in TASKS}),
        "zero_gate": _report({task: 37 for task in TASKS}),
        "shuffled": _report({task: 38 for task in TASKS}),
        "masked": _report({task: 36 for task in TASKS}),
    }

    result = apply_gate(normal, controls)

    assert result["accepted"] is True
    assert all(result["checks"].values())


def test_p1_gate_rejects_task_regression_larger_than_five_points() -> None:
    normal = _report({task: 40 for task in TASKS})
    a0_successes = {task: 38 for task in TASKS}
    a0_successes[TASKS[0]] = 44
    controls = {
        "a0": _report(a0_successes),
        "zero_gate": _report({task: 37 for task in TASKS}),
        "shuffled": _report({task: 38 for task in TASKS}),
        "masked": _report({task: 36 for task in TASKS}),
    }

    result = apply_gate(normal, controls)

    assert result["checks"]["normal_exceeds_a0"] is True
    assert result["checks"]["no_task_regression_over_5pp"] is False
    assert result["accepted"] is False


def test_p1_gate_rejects_scene_key_mismatch() -> None:
    normal = _report({task: 40 for task in TASKS})
    controls = {
        "a0": _report({task: 39 for task in TASKS}),
        "zero_gate": _report({task: 37 for task in TASKS}),
        "shuffled": _report({task: 38 for task in TASKS}),
        "masked": _report({task: 36 for task in TASKS}),
    }
    broken = deepcopy(controls["masked"])
    broken["tasks"][TASKS[0]]["cells"][0]["episode_outcomes"][0][
        "scene_seed"
    ] += 1_000_000
    controls["masked"] = broken

    try:
        apply_gate(normal, controls)
    except ValueError as error:
        assert "scene keys" in str(error)
    else:
        raise AssertionError("scene-key mismatch was not rejected")

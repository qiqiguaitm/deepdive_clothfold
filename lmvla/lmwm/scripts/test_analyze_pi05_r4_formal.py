from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_pi05_r4_formal import analyze


TASKS = [f"task_{index}" for index in range(6)]


def _write_report(
    path: Path,
    successes_per_cell: int,
    *,
    task_overrides: dict[str, int] | None = None,
) -> None:
    tasks = {}
    total_successes = 0
    for task_index, task in enumerate(TASKS):
        task_successes = (task_overrides or {}).get(task, successes_per_cell)
        cells = []
        for seed in range(4):
            outcomes = []
            for episode in range(50):
                success = episode < task_successes
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
                    "successes": task_successes,
                    "success_rate": task_successes / 50,
                    "episode_outcomes": outcomes,
                }
            )
        tasks[task] = {
            "cells": cells,
            "mean_success_rate": task_successes / 50,
            "total_episodes": 200,
            "total_successes": task_successes * 4,
        }
    macro_success_rate = sum(
        task["mean_success_rate"] for task in tasks.values()
    ) / len(tasks)
    path.write_text(
        json.dumps(
            {
                "macro_success_rate": macro_success_rate,
                "micro_success_rate": total_successes / 1200,
                "summary_count": 24,
                "task_count": 6,
                "tasks": tasks,
                "total_episodes": 1200,
                "total_successes": total_successes,
            }
        )
    )


def test_gate_accepts_terminal_arm_only_when_it_exceeds_both_controls(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "ordinary.json"
    terminal = tmp_path / "terminal.json"
    crave = tmp_path / "crave.json"
    _write_report(ordinary, 30)
    _write_report(terminal, 35)
    _write_report(crave, 32)

    result = analyze(
        ordinary,
        terminal,
        crave,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert result["accepted"] is True
    assert result["comparisons"]["ordinary"][
        "terminal_minus_control_macro"
    ] == pytest.approx(0.1)
    assert result["comparisons"]["outcome_free_crave"][
        "terminal_minus_control_macro"
    ] == pytest.approx(0.06)


def test_gate_rejects_terminal_arm_below_either_control(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary.json"
    terminal = tmp_path / "terminal.json"
    crave = tmp_path / "crave.json"
    _write_report(ordinary, 30)
    _write_report(terminal, 35)
    _write_report(crave, 36)

    result = analyze(
        ordinary,
        terminal,
        crave,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    assert result["accepted"] is False


def test_gate_rejects_macro_gain_with_task_regression_over_five_points(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "ordinary.json"
    terminal = tmp_path / "terminal.json"
    crave = tmp_path / "crave.json"
    _write_report(ordinary, 30)
    _write_report(terminal, 35, task_overrides={TASKS[0]: 26})
    _write_report(crave, 30)

    result = analyze(
        ordinary,
        terminal,
        crave,
        bootstrap_samples=20,
        bootstrap_seed=7,
    )

    assert result["terminal_macro_success_rate"] > 0.6
    assert result["comparisons"]["ordinary"][
        "terminal_minus_control_macro"
    ] > 0
    assert result["comparisons"]["ordinary"][
        "no_task_regression_below_minus_0_05"
    ] is False
    assert result["accepted"] is False

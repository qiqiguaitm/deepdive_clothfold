from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_pi05_r4_formal import analyze


TASKS = [f"task_{index}" for index in range(6)]


def _write_report(path: Path, successes_per_cell: int) -> None:
    tasks = {}
    total_successes = 0
    for task_index, task in enumerate(TASKS):
        cells = []
        for seed in range(4):
            outcomes = []
            for episode in range(50):
                success = episode < successes_per_cell
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
                    "successes": successes_per_cell,
                    "success_rate": successes_per_cell / 50,
                    "episode_outcomes": outcomes,
                }
            )
        tasks[task] = {
            "cells": cells,
            "mean_success_rate": successes_per_cell / 50,
            "total_episodes": 200,
            "total_successes": successes_per_cell * 4,
        }
    path.write_text(
        json.dumps(
            {
                "macro_success_rate": successes_per_cell / 50,
                "micro_success_rate": successes_per_cell / 50,
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

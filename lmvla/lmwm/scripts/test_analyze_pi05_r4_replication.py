from __future__ import annotations

import json
from pathlib import Path

from analyze_pi05_r4_replication import ARMS, TRAIN_SEEDS, analyze


TASKS = [f"task_{index}" for index in range(6)]


def _report(path: Path, successes: int, *, task_zero: int | None = None) -> None:
    tasks = {}
    total = 0
    for task_index, task in enumerate(TASKS):
        count = task_zero if task_index == 0 and task_zero is not None else successes
        cells = []
        for eval_seed in range(4):
            outcomes = []
            for episode in range(50):
                success = episode < count
                total += int(success)
                outcomes.append(
                    {
                        "scene_seed": task_index * 100_000 + eval_seed * 1000 + episode,
                        "success": success,
                    }
                )
            cells.append({"eval_seed": eval_seed, "episode_outcomes": outcomes})
        tasks[task] = {"cells": cells}
    path.write_text(
        json.dumps(
            {
                "summary_count": 24,
                "task_count": 6,
                "total_episodes": 1200,
                "macro_success_rate": total / 1200,
                "tasks": tasks,
            }
        )
    )


def _panel(tmp_path: Path, terminal: int = 35, *, unsafe: bool = False) -> dict:
    paths = {}
    for seed in TRAIN_SEEDS:
        for arm in ARMS:
            path = tmp_path / f"{seed}-{arm}.json"
            successes = {"ordinary": 25, "outcome_free_crave": 28, "terminal_outcome": terminal}[arm]
            _report(path, successes, task_zero=20 if unsafe and arm == "terminal_outcome" else None)
            paths[(seed, arm)] = path
    return paths


def test_accepts_replicated_positive_safe_effect(tmp_path: Path) -> None:
    result = analyze(_panel(tmp_path), bootstrap_samples=100, bootstrap_seed=7)
    assert result["accepted"] is True
    assert all(result["checks"].values())


def test_rejects_nonpositive_replication(tmp_path: Path) -> None:
    result = analyze(_panel(tmp_path, terminal=26), bootstrap_samples=50, bootstrap_seed=7)
    assert result["accepted"] is False
    assert result["checks"]["positive_95ci_vs_outcome_free_crave"] is False


def test_rejects_any_seed_task_regression_over_five_points(tmp_path: Path) -> None:
    result = analyze(_panel(tmp_path, unsafe=True), bootstrap_samples=50, bootstrap_seed=7)
    assert result["accepted"] is False
    assert result["checks"]["task_safety_vs_outcome_free_crave"] is False

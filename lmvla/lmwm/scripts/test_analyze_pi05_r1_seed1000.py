import pytest

from analyze_pi05_r1_seed1000 import analyze
from analyze_pi05_r1_seed1000 import outcomes


def report(successes: set[tuple[int, int, int]]) -> dict:
    tasks = {}
    for task_id in range(6):
        cells = []
        for eval_seed in range(4):
            episode_outcomes = [
                {
                    "scene_seed": scene_seed,
                    "success": (task_id, eval_seed, scene_seed) in successes,
                }
                for scene_seed in range(50)
            ]
            cells.append({"eval_seed": eval_seed, "episode_outcomes": episode_outcomes})
        tasks[f"task{task_id}"] = {"cells": cells}
    return {"summary_count": 24, "total_episodes": 1200, "tasks": tasks}


def success_prefix(count: int) -> set[tuple[int, int, int]]:
    keys = [
        (task, eval_seed, scene)
        for task in range(6)
        for eval_seed in range(4)
        for scene in range(50)
    ]
    return set(keys[:count])


def test_r1_gate_accepts_combined_only_when_it_beats_every_control():
    combined = report(success_prefix(960))
    controls = {
        "a0": report(success_prefix(720)),
        "predictive": report(success_prefix(740)),
        "crave": report(success_prefix(760)),
        "zero_route": report(success_prefix(700)),
        "shuffled_action": report(success_prefix(680)),
    }
    result = analyze(combined, controls)
    assert result["accepted"]
    assert all(value for value in result["checks"].values())


def test_r1_gate_rejects_content_null():
    same = report(success_prefix(720))
    result = analyze(
        same,
        {
            "a0": same,
            "predictive": same,
            "crave": same,
            "zero_route": same,
            "shuffled_action": same,
        },
    )
    assert not result["accepted"]
    assert result["comparisons"]["zero_route"]["paired_bootstrap_ci95"] == [0.0, 0.0]


def test_r1_reports_require_exact_protocol_size():
    invalid = report(set())
    invalid["total_episodes"] = 1199
    with pytest.raises(ValueError, match="1,200"):
        outcomes(invalid)

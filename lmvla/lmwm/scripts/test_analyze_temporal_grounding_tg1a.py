from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("analyze_temporal_grounding_tg1a.py")
SPEC = importlib.util.spec_from_file_location("analyze_temporal_grounding_tg1a", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


TASKS = tuple(f"task_{index}" for index in range(6))


def report(value_fn) -> dict:
    records = {}
    for eval_seed in MODULE.EVAL_SEEDS:
        for task_index, task in enumerate(TASKS):
            for scene_seed in range(50):
                records[(eval_seed, task, scene_seed)] = bool(
                    value_fn(eval_seed, task_index, scene_seed)
                )
    return {"records": records, "tasks": TASKS}


def test_tg1a_analysis_accepts_strong_content_and_preserves_task_effects() -> None:
    normal = report(lambda *_: True)
    shuffled = report(lambda _seed, task, _scene: task == 5)
    null = report(lambda *_: False)
    persistence = report(lambda _seed, _task, scene: scene % 2 == 0)
    result = MODULE.analyze(
        {
            "normal": normal,
            "shuffled": shuffled,
            "null": null,
            "persistence": persistence,
        },
        bootstrap_samples=300,
        bootstrap_seed=7,
    )

    assert result["claim_gates"]["correct_future_content_used"] is True
    assert result["claim_gates"]["future_route_necessary"] is True
    assert result["comparisons"]["shuffled"]["task_effects"]["task_5"] == 0.0
    assert result["episodes_per_condition"] == 1200


def test_tg1a_analysis_rejects_unpaired_conditions() -> None:
    normal = report(lambda *_: True)
    control = report(lambda *_: False)
    control["records"].pop(next(iter(control["records"])))
    try:
        MODULE.paired_hierarchy(normal, control)
    except ValueError as exc:
        assert "not exactly paired" in str(exc)
    else:
        raise AssertionError("unpaired reports were accepted")

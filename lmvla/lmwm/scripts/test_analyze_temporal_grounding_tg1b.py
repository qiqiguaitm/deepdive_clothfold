from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("analyze_temporal_grounding_tg1b.py")
SPEC = importlib.util.spec_from_file_location("analyze_temporal_grounding_tg1b", SOURCE)
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


def test_tg1b_positive_difference_in_differences() -> None:
    reports = {
        "future_off_e36": report(lambda *_: False),
        "future_off_e50": report(lambda *_: False),
        "local_wm_e36": report(lambda *_: False),
        "local_wm_e50": report(lambda *_: True),
    }
    result = MODULE.analyze(reports, bootstrap_samples=200, bootstrap_seed=3)
    assert result["difference_in_differences"]["mean"] == 1.0
    assert result["claim_gate"]["local_wm_specific_cadence_sensitivity"] is True
    assert len(result["difference_in_differences"]["task_effects"]) == 6


def test_tg1b_rejects_unpaired_panel() -> None:
    reports = {
        "future_off_e36": report(lambda *_: False),
        "future_off_e50": report(lambda *_: False),
        "local_wm_e36": report(lambda *_: False),
        "local_wm_e50": report(lambda *_: True),
    }
    reports["local_wm_e50"]["records"].pop(next(iter(reports["local_wm_e50"]["records"])))
    try:
        MODULE.analyze(reports, bootstrap_samples=10)
    except ValueError as exc:
        assert "not exactly paired" in str(exc)
    else:
        raise AssertionError("unpaired TG1B panel was accepted")

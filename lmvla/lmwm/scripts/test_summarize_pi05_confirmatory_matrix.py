from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("summarize_pi05_confirmatory_matrix.py")
SPEC = importlib.util.spec_from_file_location("pi05_confirmatory_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_fixture() -> tuple[dict, dict]:
    tasks = ["task_a", "task_b"]
    manifest = {
        "tasks": tasks,
        "episodes_per_cell": 3,
        "eval_seeds": {
            "0": {"task_a": [10, 11, 12], "task_b": [20, 21, 22]},
            "1": {"task_a": [30, 31, 32], "task_b": [40, 41, 42]},
        },
    }
    report_tasks = {}
    for task in tasks:
        cells = []
        for eval_seed in (0, 1):
            scene_seeds = manifest["eval_seeds"][str(eval_seed)][task]
            outcomes = [
                {"episode_id": index, "scene_seed": seed, "success": index % 2 == 0}
                for index, seed in enumerate(scene_seeds)
            ]
            cells.append(
                {
                    "eval_seed": eval_seed,
                    "episodes": 3,
                    "successes": 2,
                    "success_rate": 2 / 3,
                    "episode_outcomes": outcomes,
                }
            )
        report_tasks[task] = {"cells": cells}
    report = {
        "summary_count": 4,
        "task_count": 2,
        "total_episodes": 12,
        "tasks": report_tasks,
    }
    return report, manifest


def test_audit_accepts_exact_manifest() -> None:
    report, manifest = make_fixture()
    audit = MODULE.audit_report(report, manifest)
    assert audit["accepted"]
    assert audit["checked_cells"] == 4


def test_audit_rejects_scene_seed_mutation() -> None:
    report, manifest = make_fixture()
    report = copy.deepcopy(report)
    report["tasks"]["task_a"]["cells"][0]["episode_outcomes"][0]["scene_seed"] += 1
    audit = MODULE.audit_report(report, manifest)
    assert not audit["accepted"]
    assert any("scene seed order mismatch" in error for error in audit["errors"])


def test_audit_rejects_missing_episode() -> None:
    report, manifest = make_fixture()
    report = copy.deepcopy(report)
    report["tasks"]["task_a"]["cells"][0]["episode_outcomes"].pop()
    audit = MODULE.audit_report(report, manifest)
    assert not audit["accepted"]
    assert any("outcome_count=2" in error for error in audit["errors"])


def _contrast_report(success: bool) -> dict:
    tasks = {}
    for task_index in range(6):
        task = f"task_{task_index}"
        tasks[task] = {
            "cells": [
                {
                    "eval_seed": 0,
                    "episode_outcomes": [
                        {
                            "scene_seed": task_index * 100 + episode,
                            "success": success,
                        }
                        for episode in range(4)
                    ],
                }
            ]
        }
    return {"tasks": tasks}


def test_hierarchical_contrast_resamples_training_seeds_before_episodes() -> None:
    baseline = {seed: _contrast_report(False) for seed in (1000, 1001, 1002)}
    candidate = {
        1000: _contrast_report(True),
        1001: _contrast_report(False),
        1002: _contrast_report(False),
    }
    result = MODULE.paired_hierarchical_contrast(
        candidate,
        baseline,
        bootstrap_samples=5_000,
        bootstrap_seed=20260802,
    )
    assert result["available"]
    assert result["paired_episode_count"] == 72
    assert result["unmatched_episode_key_count"] == 0
    assert result["point_estimate_macro_delta"] == 1 / 3
    assert result["ci95"] == pytest.approx([0.0, 1.0])
    assert result["hierarchy"] == (
        "resample training seeds, then paired episodes within each task"
    )

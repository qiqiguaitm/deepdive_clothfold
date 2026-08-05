from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("analyze_mt6_scope.py")
SPEC = importlib.util.spec_from_file_location("analyze_mt6_scope", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


TASKS = [
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
]


def manifest() -> dict:
    return {
        "tasks": TASKS,
        "episodes_per_cell": 4,
        "eval_seeds": {
            "0": {
                task: [task_index * 100 + episode for episode in range(4)]
                for task_index, task in enumerate(TASKS)
            }
        },
    }


def report(multistage_success: bool) -> dict:
    frozen = manifest()
    tasks = {}
    for task in TASKS:
        success = multistage_success and task in TASKS[:3]
        outcomes = [
            {"episode_id": index, "scene_seed": seed, "success": success}
            for index, seed in enumerate(frozen["eval_seeds"]["0"][task])
        ]
        tasks[task] = {
            "cells": [
                {
                    "eval_seed": 0,
                    "episodes": 4,
                    "successes": 4 if success else 0,
                    "success_rate": 1.0 if success else 0.0,
                    "episode_outcomes": outcomes,
                }
            ]
        }
    return {
        "summary_count": 6,
        "task_count": 6,
        "total_episodes": 24,
        "tasks": tasks,
    }


def scope() -> dict:
    return {
        "version": "test-scope",
        "groups": {
            "multistage_aliasing": TASKS[:3],
            "reactive_geometric_control": TASKS[3:],
        },
        "statistics": {"bootstrap_samples": 500, "bootstrap_seed": 7},
        "interpretation": "test",
    }


def test_scope_interaction_uses_training_seed_then_episode_hierarchy() -> None:
    candidate = {seed: report(True) for seed in MODULE.EXPECTED_SEEDS}
    baseline = {seed: report(False) for seed in MODULE.EXPECTED_SEEDS}
    result = MODULE.analyze(candidate, baseline, manifest(), scope())
    assert result["complete"]
    assert result["paired_episodes"] == 72
    assert result["point_estimate"] == {
        "multistage_aliasing": 1.0,
        "reactive_geometric_control": 0.0,
        "scope_interaction": 1.0,
    }
    assert result["ci95"]["scope_interaction"] == pytest.approx([1.0, 1.0])
    assert result["scope_interaction_ci95_excludes_zero"]


def test_scope_groups_must_partition_frozen_manifest() -> None:
    invalid = copy.deepcopy(scope())
    invalid["groups"]["reactive_geometric_control"][0] = "handover_block"
    with pytest.raises(ValueError, match="overlap"):
        MODULE.validate_scope(invalid, manifest())

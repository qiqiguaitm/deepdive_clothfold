from __future__ import annotations

import pytest

import analyze_temporal_grounding_tg4 as analyzer


def make_panels() -> dict:
    tasks = ("task_a", "task_b")
    values = {
        "clean_base_normal": 0.0,
        "future_off_normal": 1.0,
        "auxiliary_only_normal": 1.0,
        "conditioning_only_normal": 0.0,
        "parameter_matched_null_normal": 0.0,
        "full_normal": 1.0,
        "full_shuffled": 0.0,
    }
    panels = {}
    for name, value in values.items():
        panels[name] = {}
        for train_seed in analyzer.TRAIN_SEEDS:
            records = {}
            for eval_seed in analyzer.EVAL_SEEDS:
                for task in tasks:
                    for episode in range(5):
                        records[(eval_seed, task, episode)] = value
            panels[name][train_seed] = {"tasks": tasks, "records": records}
    return panels


def test_analyze_uses_training_seed_as_highest_level() -> None:
    result = analyzer.analyze(make_panels(), bootstrap_samples=1_000)

    assert result["complete"]
    assert result["holm_family"] == [name for name, _ in analyzer.COMPARISONS]
    pretraining = result["comparisons"]["pretraining"]
    assert pretraining["mean_effect"] == 1.0
    assert pretraining["hierarchical_ci95"] == [1.0, 1.0]
    assert pretraining["task_safety_passed"]
    assert pretraining["bootstrap"]["levels"][0] == "training_seed"
    assert pretraining["accepted"]
    assert result["comparisons"]["content_use"]["accepted"]


def test_analyze_rejects_missing_panel() -> None:
    panels = make_panels()
    panels.pop("full_shuffled")
    with pytest.raises(ValueError, match="Panel set mismatch"):
        analyzer.analyze(panels, bootstrap_samples=10)

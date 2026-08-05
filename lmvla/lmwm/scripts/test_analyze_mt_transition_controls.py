import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_mt_transition_controls.py")
SPEC = importlib.util.spec_from_file_location("analyze_mt_transition_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
analyze = MODULE.analyze


def report(outcomes):
    cells = []
    for eval_seed, values in enumerate(outcomes):
        cells.append(
            {
                "eval_seed": eval_seed,
                "episode_outcomes": [
                    {"scene_seed": 100 + index, "success": success}
                    for index, success in enumerate(values)
                ],
            }
        )
    rate = sum(sum(values) for values in outcomes) / sum(len(values) for values in outcomes)
    return {"macro_success_rate": rate, "tasks": {"task": {"cells": cells}}}


def test_paired_mcnemar_and_holm_are_reported():
    correct = report([[1, 1, 1, 0], [1, 1, 0, 0]])
    null = report([[0, 0, 1, 0], [0, 1, 0, 0]])
    shuffled = report([[0, 1, 0, 0], [1, 0, 0, 0]])
    result = analyze(correct, {"null": null, "within_task": shuffled})
    assert result["holm_family_size"] == 4
    assert result["holm_family_definition"].startswith("all pooled and task-level")
    assert all("holm_adjusted_p" in row for row in result["tests"])
    assert all("candidate_success_rate" in row for row in result["tests"])
    assert all("control_success_rate" in row for row in result["tests"])
    assert all(row["success_rate_delta"] >= 0 for row in result["tests"])


def test_task_level_effect_size_matches_paired_success_rates():
    correct = report([[1, 1, 0, 0]])
    null = report([[1, 0, 0, 0]])
    result = analyze(correct, {"null": null})
    for row in result["tests"]:
        assert row["candidate_success_rate"] == 0.5
        assert row["control_success_rate"] == 0.25
        assert row["success_rate_delta"] == 0.25
        assert row["candidate_only_success"] == 1
        assert row["control_only_success"] == 0


def test_atomic_write_replaces_complete_file(tmp_path):
    output = tmp_path / "nested" / "result.json"
    MODULE.atomic_write_text(output, '{"first": true}\n')
    MODULE.atomic_write_text(output, '{"second": true}\n')
    assert output.read_text() == '{"second": true}\n'
    assert not list(output.parent.glob(".*.tmp"))

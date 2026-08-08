import json
from pathlib import Path

import pytest

from audit_temporal_grounding_retry_panel import read_retry_cap, validate_summary


def test_read_retry_cap_requires_one_literal_export(tmp_path: Path) -> None:
    runner = tmp_path / "runner.sh"
    runner.write_text("export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=3\n")
    assert read_retry_cap(runner) == 3

    runner.write_text("echo no-cap\n")
    with pytest.raises(ValueError, match="expected one fixed retry cap"):
        read_retry_cap(runner)


def test_validate_summary_requires_exact_scene_order(tmp_path: Path) -> None:
    summary = tmp_path / "seed0/checkpoint/run/tasks/task_a/summary.json"
    summary.parent.mkdir(parents=True)
    manifest = {"eval_seeds": {"0": {"task_a": list(range(50))}}}
    payload = {
        "task_name": "task_a",
        "n_episodes": 50,
        "fixed_seed_manifest": {
            "sha256": "scene-hash",
            "eval_seed": 0,
            "task_name": "task_a",
            "count": 50,
        },
        "episodes": [{"seed": seed} for seed in range(50)],
    }
    summary.write_text(json.dumps(payload))

    assert validate_summary(summary, manifest, "scene-hash") == (0, "task_a")

    payload["episodes"][0]["seed"] = 99
    summary.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="exact_ordered_scene_ids"):
        validate_summary(summary, manifest, "scene-hash")

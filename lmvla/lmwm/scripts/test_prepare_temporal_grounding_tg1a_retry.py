import json
from pathlib import Path

import pytest

from prepare_temporal_grounding_tg1a_retry import (
    SCHEMA_ERROR,
    archive_failed_runtime_v4,
)


def _make_failed_run(repo: Path, condition: str) -> Path:
    root = (
        repo
        / "lmvla/lawam/results/eval_runs/robotwin"
        / f"temporal_grounding_tg1a_{condition}"
    )
    for seed in range(4):
        run = root / f"seed{seed}" / "checkpoint" / f"tg1a-{condition}-seed{seed}"
        status_dir = run / ".task_status"
        status_dir.mkdir(parents=True)
        for task in range(6):
            (status_dir / f"task{task}.json").write_text(
                json.dumps({"status": "failed"})
            )
        server_log = run / "workers/worker_0/server.log"
        server_log.parent.mkdir(parents=True)
        server_log.write_text(f"KeyError: {SCHEMA_ERROR}\n")
    return root


def test_archives_only_complete_schema_failure(tmp_path: Path) -> None:
    root = _make_failed_run(tmp_path, "null")

    archive = archive_failed_runtime_v4(tmp_path, "null")

    assert archive == root.with_name(root.name + ".failed_runtime_v4_schema")
    assert archive.is_dir()
    assert not root.exists()


def test_check_only_validates_without_archiving(tmp_path: Path) -> None:
    root = _make_failed_run(tmp_path, "null")

    archive = archive_failed_runtime_v4(tmp_path, "null", check_only=True)

    assert archive == root.with_name(root.name + ".failed_runtime_v4_schema")
    assert root.is_dir()
    assert not archive.exists()


def test_refuses_any_existing_summary(tmp_path: Path) -> None:
    root = _make_failed_run(tmp_path, "persistence")
    summary = next(root.glob("seed*/**/.task_status")).parent / "summary.json"
    summary.write_text("{}\n")

    with pytest.raises(RuntimeError, match="found 1 summaries"):
        archive_failed_runtime_v4(tmp_path, "persistence")


def test_normal_archives_only_empty_feature_root(tmp_path: Path) -> None:
    _make_failed_run(tmp_path, "normal")
    feature_root = tmp_path / "logs/temporal_grounding/tg1a/predicted_endpoint_features"
    feature_root.mkdir(parents=True)

    archive_failed_runtime_v4(tmp_path, "normal")

    assert feature_root.with_name(
        feature_root.name + ".failed_runtime_v4_schema"
    ).is_dir()


def test_refuses_wrong_failure_cause(tmp_path: Path) -> None:
    root = _make_failed_run(tmp_path, "null")
    next(root.glob("seed*/**/server.log")).write_text("different error\n")

    with pytest.raises(RuntimeError, match="matches=3"):
        archive_failed_runtime_v4(tmp_path, "null")

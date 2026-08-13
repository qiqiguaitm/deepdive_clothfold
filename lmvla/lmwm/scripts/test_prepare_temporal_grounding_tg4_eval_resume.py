import json
from pathlib import Path

import pytest

from prepare_temporal_grounding_tg4_eval_resume import (
    TASKS,
    expected_run_dir,
    validate_resume_root,
)


def make_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    root = tmp_path / "result"
    run = expected_run_dir(
        root,
        arm="full",
        train_seed=1100,
        condition="normal",
        eval_seed=0,
    )
    run.mkdir(parents=True)
    (run / "run_meta.json").write_text(
        json.dumps(
            {
                "run_group": run.parent.name,
                "run_tag": run.name,
                "checkpoint_path": str(checkpoint),
                "checkpoint_alias": "tg4_full_s1100",
                "task_config": "demo_clean",
                "requested_tasks": TASKS,
                "expected_test_num": 50,
            }
        )
    )
    return root, checkpoint, run


def test_accepts_exact_existing_run(tmp_path: Path) -> None:
    root, checkpoint, run = make_run(tmp_path)
    assert validate_resume_root(
        root,
        checkpoint,
        arm="full",
        train_seed=1100,
        condition="normal",
        eval_seed=0,
    ) == run
    assert (
        validate_resume_root(
            root,
            checkpoint,
            arm="full",
            train_seed=1100,
            condition="normal",
            eval_seed=1,
        )
        is None
    )


def test_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    root, _, _ = make_run(tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(ValueError, match="checkpoint_path"):
        validate_resume_root(
            root,
            other,
            arm="full",
            train_seed=1100,
            condition="normal",
            eval_seed=0,
        )


def test_rejects_protocol_mismatch(tmp_path: Path) -> None:
    root, checkpoint, run = make_run(tmp_path)
    meta = json.loads((run / "run_meta.json").read_text())
    meta["expected_test_num"] = 1
    (run / "run_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="expected_test_num"):
        validate_resume_root(
            root,
            checkpoint,
            arm="full",
            train_seed=1100,
            condition="normal",
            eval_seed=0,
        )


def test_rejects_unexpected_result_entry(tmp_path: Path) -> None:
    root, checkpoint, _ = make_run(tmp_path)
    (root / "foreign").mkdir()
    with pytest.raises(ValueError, match="unexpected entry"):
        validate_resume_root(
            root,
            checkpoint,
            arm="full",
            train_seed=1100,
            condition="normal",
            eval_seed=0,
        )

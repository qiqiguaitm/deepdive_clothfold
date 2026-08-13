from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import finalize_temporal_grounding_tg4_todo as finalizer


def inputs(tmp_path: Path) -> dict[str, Path]:
    rows = {
        name: {
            "accepted": index % 2 == 0,
            "mean_effect": 0.1 - index * 0.01,
            "hierarchical_ci95": [0.01, 0.2],
            "holm_adjusted_p": 0.01,
            "minimum_training_seed_task_effect": -0.01,
            "task_safety_passed": True,
        }
        for index, name in enumerate(finalizer.COMPARISONS)
    }
    report = tmp_path / "result.json"
    report.write_text(
        json.dumps(
            {
                "protocol": finalizer.PROTOCOL,
                "complete": True,
                "holm_family": list(finalizer.COMPARISONS),
                "comparisons": rows,
            }
        ),
        encoding="utf-8",
    )
    analysis_marker = tmp_path / "analysis.ok"
    analysis_marker.write_text(
        "validated=true\n"
        f"protocol={finalizer.PROTOCOL}\n"
        + "".join(
            f"{name}={str(row['accepted']).lower()}\n" for name, row in rows.items()
        ),
        encoding="utf-8",
    )
    todo = tmp_path / "PAPER_TODO.md"
    todo.write_text(
        "# TODO\n\nUpdated: 2026-08-12 00:00 UTC\n\n"
        "- [ ] **TG4-T01--T18 [ACTIVE; 8/18 COMPLETE]** body\n"
        "- [ ] **TG4-I1 [BLOCKED by T01--T18]** body\n"
        "- [ ] **TG4-E1 [IMPLEMENTED; BLOCKED by I1]** body\n"
        "- [ ] **TG4-A1 [IMPLEMENTED; BLOCKED by E1]** body\n\n"
        "### TG4 claim gates\n",
        encoding="utf-8",
    )
    return {
        "report_path": report,
        "analysis_marker": analysis_marker,
        "todo_path": todo,
        "summary_path": tmp_path / "RESULTS.md",
        "completion_marker": tmp_path / "finalized.ok",
        "lock_path": tmp_path / "finalize.lock",
    }


def test_finalize_updates_only_frozen_gates_and_writes_evidence(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    finalizer.finalize(
        **paths, now=datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)
    )

    todo = paths["todo_path"].read_text(encoding="utf-8")
    assert "Updated: 2026-08-13 08:00 UTC" in todo
    assert todo.count("- [x] **TG4-") == 4
    assert "TG4-T01--T18 [COMPLETE; 18/18 COMPLETE]" in todo
    assert todo.count(finalizer.SUMMARY_START) == 1
    assert "`pretraining`" in todo
    assert "`auxiliary_shaping`" in todo
    assert "| `content_use` |" in paths["summary_path"].read_text(encoding="utf-8")
    marker = paths["completion_marker"].read_text(encoding="utf-8")
    assert "validated=true" in marker
    assert "todo_sha256=" in marker

    finalizer.finalize(
        **paths, now=datetime(2026, 8, 13, 8, 1, tzinfo=timezone.utc)
    )
    assert paths["todo_path"].read_text(encoding="utf-8").count(
        finalizer.SUMMARY_START
    ) == 1


def test_finalize_rejects_marker_mismatch_without_writes(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    paths["analysis_marker"].write_text(
        paths["analysis_marker"].read_text(encoding="utf-8").replace(
            "pretraining=true", "pretraining=false"
        ),
        encoding="utf-8",
    )
    original = paths["todo_path"].read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="marker verdict mismatch"):
        finalizer.finalize(**paths)

    assert paths["todo_path"].read_text(encoding="utf-8") == original
    assert not paths["summary_path"].exists()
    assert not paths["completion_marker"].exists()


def test_finalize_rejects_missing_todo_gate(tmp_path: Path) -> None:
    paths = inputs(tmp_path)
    paths["todo_path"].write_text(
        paths["todo_path"].read_text(encoding="utf-8").replace("TG4-I1", "TG4-X1"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one TODO gate for TG4-I1"):
        finalizer.finalize(**paths)

    assert not paths["completion_marker"].exists()

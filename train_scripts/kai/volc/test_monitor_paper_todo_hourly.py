import json
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("monitor_paper_todo_hourly.py")
SPEC = importlib.util.spec_from_file_location("monitor_paper_todo_hourly", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


NOW = datetime(2026, 8, 10, 14, 0, tzinfo=timezone.utc)


def write_analysis_artifacts(tmp_path: Path) -> None:
    for spec in monitor.ANALYSIS_ARTIFACT_SPECS.values():
        report = tmp_path / spec["report"]
        marker = tmp_path / spec["marker"]
        report.parent.mkdir(parents=True, exist_ok=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"protocol": spec["protocol"]}), encoding="utf-8")
        marker.write_text("validated=true\n", encoding="utf-8")


def write_inputs(tmp_path: Path, *, status: str = "completed") -> tuple[Path, Path, Path]:
    todo = tmp_path / "PAPER_TODO.md"
    todo.write_text("# TODO\n- [ ] active\n## 1. History\n- [ ] old\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "timestamp": monitor.isoformat(NOW),
                "resources": {
                    "beijing": {"backup": {}},
                    "Robot-East-H20": {"watched_tasks": {}},
                    "local": {},
                },
                "queue_inventory": {},
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "tasks": {
                    task_id: {"status": status, "attempts": []}
                    for task_id in monitor.EXPECTED_TASK_IDS
                }
                | {
                    spec["task_id"]: {"status": status, "attempts": []}
                    for spec in monitor.ANALYSIS_ARTIFACT_SPECS.values()
                }
            }
        ),
        encoding="utf-8",
    )
    write_analysis_artifacts(tmp_path)
    return todo, snapshot, state


def test_frozen_completion_set_has_all_tg1a_tg1b_and_tg2r_cells() -> None:
    assert len(monitor.EXPECTED_TASK_IDS) == 37
    assert "temporal_grounding_tg1a_shuffled_eval" in monitor.EXPECTED_TASK_IDS
    assert (
        "temporal_grounding_tg1b_local_wm_e50_eval" in monitor.EXPECTED_TASK_IDS
    )
    assert "temporal_grounding_tg2r_training_integrity" in monitor.EXPECTED_TASK_IDS
    assert "temporal_grounding_tg2r_raw_milestone_seed1002_eval" in monitor.EXPECTED_TASK_IDS


def test_next_interval_boundary_aligns_hourly_poll_to_top_of_hour() -> None:
    now = datetime(2026, 8, 11, 12, 21, 38, tzinfo=timezone.utc)
    assert monitor.next_interval_boundary(now, 3600) == datetime(
        2026, 8, 11, 13, 0, tzinfo=timezone.utc
    )


def test_next_interval_boundary_advances_from_exact_boundary() -> None:
    assert monitor.next_interval_boundary(NOW, 3600) == NOW + timedelta(hours=1)


def test_collect_marks_exact_completed_set_complete(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    assert record["complete"] is True
    assert record["monitor_status"] == "complete"
    assert record["task_summary"]["status_counts"] == {"completed": 37}
    assert record["todo"]["unchecked_current_override"] == 1
    assert record["todo"]["unchecked_total"] == 2
    assert record["final_analyses"]["complete"] is True


def test_collect_requires_final_analysis_artifacts(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    (tmp_path / monitor.ANALYSIS_ARTIFACT_SPECS["tg2"]["marker"]).unlink()
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    assert record["complete"] is False
    assert record["monitor_status"] == "active"
    assert record["final_analyses"]["analyses"]["tg2"]["status"] == "partial"


def test_collect_requires_registered_completed_analysis_task(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    payload = json.loads(state.read_text())
    del payload["tasks"][monitor.ANALYSIS_ARTIFACT_SPECS["tg2"]["task_id"]]
    state.write_text(json.dumps(payload), encoding="utf-8")
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    assert record["complete"] is False
    assert record["monitor_status"] == "active"
    assert record["final_analyses"]["analyses"]["tg2"]["status"] == "unregistered"


def test_collect_rejects_invalid_final_analysis_artifact(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    report = tmp_path / monitor.ANALYSIS_ARTIFACT_SPECS["tg2"]["report"]
    report.write_text(json.dumps({"protocol": "wrong"}), encoding="utf-8")
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    assert record["complete"] is False
    assert record["monitor_status"] == "degraded"
    assert record["final_analyses"]["analyses"]["tg2"]["status"] == "invalid"


def test_collect_does_not_treat_missing_or_disabled_task_as_complete(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    payload = json.loads(state.read_text())
    missing = monitor.EXPECTED_TASK_IDS[0]
    disabled = monitor.EXPECTED_TASK_IDS[1]
    del payload["tasks"][missing]
    payload["tasks"][disabled]["status"] = "disabled"
    state.write_text(json.dumps(payload), encoding="utf-8")
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    assert record["complete"] is False
    assert missing in record["task_summary"]["missing"]
    assert disabled in record["task_summary"]["incomplete"]


def test_collect_accepts_validated_tg2r_scientific_rejection(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    tg2 = monitor.ANALYSIS_ARTIFACT_SPECS["tg2"]
    (tmp_path / tg2["report"]).unlink()
    (tmp_path / tg2["marker"]).unlink()
    rejection = tmp_path / tg2["rejection_report"]
    rejection.parent.mkdir(parents=True, exist_ok=True)
    rejection.write_text(
        json.dumps(
            {
                "protocol": tg2["rejection_protocol"],
                "accepted_for_evaluation": False,
                "scientific_disposition": {"evaluations_retired": 9},
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(state.read_text())
    for task_id, task in payload["tasks"].items():
        if task_id.startswith("temporal_grounding_tg2r_"):
            task["status"] = "disabled"
            task["disabled_reason"] = f"scientific gate rejected: {rejection}"
    payload["tasks"][tg2["task_id"]]["status"] = "disabled"
    state.write_text(json.dumps(payload), encoding="utf-8")

    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )

    assert record["complete"] is True
    assert record["final_analyses"]["analyses"]["tg2"]["status"] == "rejected"
    assert not any(
        task_id.startswith("temporal_grounding_tg2r_")
        for task_id in record["task_summary"]["incomplete"]
    )


def test_collect_rejects_tg2_result_beside_rejection_decision(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    tg2 = monitor.ANALYSIS_ARTIFACT_SPECS["tg2"]
    rejection = tmp_path / tg2["rejection_report"]
    rejection.parent.mkdir(parents=True, exist_ok=True)
    rejection.write_text(
        json.dumps(
            {
                "protocol": tg2["rejection_protocol"],
                "accepted_for_evaluation": False,
                "scientific_disposition": {"evaluations_retired": 9},
            }
        ),
        encoding="utf-8",
    )

    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )

    tg2_analysis = record["final_analyses"]["analyses"]["tg2"]
    assert tg2_analysis["status"] == "invalid"
    assert tg2_analysis["validated"] is False


def test_stale_scheduler_snapshot_is_degraded(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["timestamp"] = monitor.isoformat(NOW - timedelta(hours=2))
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        max_snapshot_age_seconds=300,
        repo_path=tmp_path,
    )
    assert record["complete"] is False
    assert record["monitor_status"] == "degraded"
    assert not record["scheduler"]["healthy"]


def test_collect_reports_task_transition(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path, status="pending")
    first = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )
    payload = json.loads(state.read_text())
    task_id = monitor.EXPECTED_TASK_IDS[0]
    payload["tasks"][task_id] = {
        "status": "running",
        "attempts": [{"last_state": "Running", "job_id": "t-test"}],
    }
    state.write_text(json.dumps(payload), encoding="utf-8")
    second = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        previous=first,
        now=NOW,
        repo_path=tmp_path,
    )
    assert second["transitions"] == [
        {
            "task_id": task_id,
            "before": ["pending", None, None],
            "after": ["running", "Running", "t-test"],
        }
    ]


def test_collect_and_markdown_include_active_evaluation_progress(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path, status="pending")
    payload = json.loads(state.read_text())
    task_id = "temporal_grounding_tg1a_persistence_eval"
    payload["tasks"][task_id] = {
        "status": "running",
        "attempts": [{"last_state": "Running", "job_id": "t-test"}],
        "runtime_progress": "episodes=1152/1200",
        "runtime_progress_changed_at": "2026-08-10T13:59:50Z",
        "artifact_progress": "completion artifacts east=20/24, north=0/24",
        "artifact_stale_seconds": 0,
    }
    state.write_text(json.dumps(payload), encoding="utf-8")

    record = monitor.collect(
        todo_path=todo,
        snapshot_path=snapshot,
        state_path=state,
        now=NOW,
        repo_path=tmp_path,
    )

    row = record["tasks"][task_id]
    assert row["runtime_progress"] == "episodes=1152/1200"
    assert row["artifact_progress"] == (
        "completion artifacts east=20/24, north=0/24"
    )
    assert row["progress_changed_at"] == "2026-08-10T13:59:50Z"
    assert row["progress_stale_seconds"] == 0
    markdown = monitor.render_markdown(record)
    assert "## Active Evaluations" in markdown
    assert "episodes=1152/1200" in markdown
    assert "completion artifacts east=20/24, north=0/24" in markdown


def test_heartbeat_prefers_running_location_and_filters_inactive_tasks() -> None:
    active_id = "temporal_grounding_tg2r_raw_milestone_seed1000_train"
    inactive_id = "temporal_grounding_tg2r_fixed_endpoint_seed1000_train"
    tasks = {
        active_id: {"status": "running", "execution_state": "Running"},
        inactive_id: {"status": "completed", "execution_state": "Completed"},
    }
    snapshot = {
        "resources": {
            "Robot-East-H20": {
                "watched_tasks": {
                    "tg2r_raw_milestone_seed1000": {
                        "status": "RUNNING",
                        "step": 18000,
                    },
                    "tg2r_fixed_endpoint_seed1000": {
                        "status": "STALE_LOG",
                        "step": 20000,
                    },
                }
            },
            "beijing": {
                "watched_tasks": {
                    "tg2r_raw_milestone_seed1000": {"status": "WAITING_FOR_LOG"}
                }
            },
        }
    }
    result = monitor.heartbeat_metrics(snapshot, tasks)
    assert result == {
        "tg2r_raw_milestone_seed1000": {
            "resource": "Robot-East-H20",
            "status": "RUNNING",
            "step": 18000,
        }
    }


def test_once_writes_all_monitor_artifacts(tmp_path: Path) -> None:
    todo, snapshot, state = write_inputs(tmp_path, status="pending")
    jsonl = tmp_path / "monitor.jsonl"
    latest_json = tmp_path / "latest.json"
    latest_md = tmp_path / "latest.md"
    result = monitor.main(
        [
            "--once",
            "--max-snapshot-age-seconds",
            "172800",
            "--todo",
            str(todo),
            "--snapshot",
            str(snapshot),
            "--state",
            str(state),
            "--jsonl",
            str(jsonl),
            "--latest-json",
            str(latest_json),
            "--latest-md",
            str(latest_md),
            "--lock",
            str(tmp_path / "monitor.lock"),
            "--repo",
            str(tmp_path),
        ]
    )
    assert result == 0
    assert len(jsonl.read_text().splitlines()) == 1
    assert json.loads(latest_json.read_text())["monitor_status"] == "active"
    assert "Paper TODO Hourly Monitor" in latest_md.read_text()

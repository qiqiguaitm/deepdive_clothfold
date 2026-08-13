#!/usr/bin/env python3
"""Persist an hourly, read-only audit of the active paper TODO.

Experiment execution remains owned by resource_aware_scheduler.py. This
monitor only reads its state/snapshot and canonical outputs, records progress,
and exits when the frozen TG4 matrix and final analysis are finished.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
DEFAULT_TODO = REPO / "lmvla/paper_iclr_lmvla/PAPER_TODO.md"
DEFAULT_SNAPSHOT = REPO / "logs/resource_scheduler_snapshot.json"
DEFAULT_STATE = REPO / "logs/resource_scheduler_state.json"
DEFAULT_JSONL = REPO / "logs/paper_todo_hourly_monitor.jsonl"
DEFAULT_LATEST_JSON = REPO / "logs/paper_todo_hourly_monitor_latest.json"
DEFAULT_LATEST_MD = REPO / "logs/paper_todo_hourly_monitor_latest.md"
DEFAULT_LOCK = REPO / "logs/paper_todo_hourly_monitor.lock"

ANALYSIS_ARTIFACT_SPECS = {
    "tg4": {
        "task_id": "temporal_grounding_tg4_analysis",
        "report": "lmvla/paper_iclr_lmvla/RESULTS_temporal_grounding_tg4.json",
        "marker": "logs/resource_markers/temporal_grounding_tg4_analysis.ok",
        "protocol": "temporal_grounding_tg4_source_decomposition_analysis_v1",
    },
}

STOP_REQUESTED = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def next_interval_boundary(now: datetime, interval_seconds: int) -> datetime:
    """Return the next epoch-aligned UTC polling boundary."""
    epoch = now.astimezone(timezone.utc).timestamp()
    next_epoch = (math.floor(epoch / interval_seconds) + 1) * interval_seconds
    return datetime.fromtimestamp(next_epoch, tz=timezone.utc)


def write_lock_status(
    handle: Any, *, started: datetime, last_poll: datetime | None, next_poll: datetime | None
) -> None:
    fields = [f"pid={os.getpid()}", f"started={isoformat(started)}"]
    if last_poll is not None:
        fields.append(f"last_poll={isoformat(last_poll)}")
    if next_poll is not None:
        fields.append(f"next_poll={isoformat(next_poll)}")
    handle.seek(0)
    handle.truncate()
    handle.write(" ".join(fields) + "\n")
    handle.flush()


def expected_task_ids() -> tuple[str, ...]:
    arms = (
        "clean_base",
        "future_off",
        "auxiliary_only",
        "conditioning_only",
        "parameter_matched_null",
        "full",
    )
    tasks = []
    for arm in arms:
        for seed in (1100, 1101, 1102):
            parent = f"temporal_grounding_tg4_{arm}_seed{seed}_train"
            tasks.extend((parent, f"{parent}_materialize_north"))
            normal_eval = f"temporal_grounding_tg4_{arm}_seed{seed}_normal_eval"
            tasks.extend((normal_eval, f"{normal_eval}_materialize_north"))
            if arm == "full":
                tasks.append(
                    f"temporal_grounding_tg4_{arm}_seed{seed}_shuffled_eval"
                )
    tasks.extend(
        (
            "temporal_grounding_tg4_training_integrity",
            "temporal_grounding_tg4_eval_north_stage",
            "temporal_grounding_tg4_analysis",
        )
    )
    return tuple(tasks)


EXPECTED_TASK_IDS = expected_task_ids()
AUXILIARY_TASK_IDS = (
    "temporal_grounding_tg4_north_stage",
    "temporal_grounding_tg4_conditioning_ddp_repair_north_stage",
)
TODO_COMPLETION_ITEMS = ("TG4-T01--T18", "TG4-I1", "TG4-E1", "TG4-A1")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def todo_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    current = text.split("## 1.", 1)[0]
    completion_items = {}
    for label in TODO_COMPLETION_ITEMS:
        match = re.search(
            rf"^- \[([ xX])\] \*\*{re.escape(label)}(?:\s|\[)",
            text,
            flags=re.MULTILINE,
        )
        completion_items[label] = (
            "missing"
            if match is None
            else ("checked" if match.group(1).lower() == "x" else "unchecked")
        )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "unchecked_current_override": current.count("- [ ]"),
        "unchecked_total": text.count("- [ ]"),
        "checked_total": text.count("- [x]") + text.count("- [X]"),
        "completion_items": completion_items,
        "completion_synced": all(
            status == "checked" for status in completion_items.values()
        ),
    }


def analysis_artifact_metrics(
    repo_path: Path, state_tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    analyses: dict[str, Any] = {}
    for name, spec in ANALYSIS_ARTIFACT_SPECS.items():
        report_path = repo_path / spec["report"]
        marker_path = repo_path / spec["marker"]
        present = {
            "report": report_path.is_file(),
            "marker": marker_path.is_file(),
        }
        row: dict[str, Any] = {
            "task_id": spec["task_id"],
            "task": task_record(state_tasks.get(spec["task_id"])),
            "report": str(report_path),
            "marker": str(marker_path),
            "expected_protocol": spec["protocol"],
            "present": present,
            "status": "missing" if not any(present.values()) else "partial",
            "validated": False,
            "error": None,
        }
        task_status = row["task"]["status"]
        if task_status == "missing":
            row["status"] = "unregistered"
        elif task_status != "completed":
            row["status"] = "task_incomplete"
        if all(present.values()):
            try:
                report = load_json(report_path)
                observed_protocol = report.get("protocol")
                marker_lines = {
                    line.strip()
                    for line in marker_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
                if observed_protocol != spec["protocol"]:
                    raise ValueError(
                        f"protocol {observed_protocol!r} != {spec['protocol']!r}"
                    )
                if "validated=true" not in marker_lines:
                    raise ValueError("marker does not contain validated=true")
                if task_status == "completed":
                    row.update(status="validated", validated=True)
            except Exception as exc:
                row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        analyses[name] = row
    return {
        "complete": all(row["validated"] for row in analyses.values()),
        "analyses": analyses,
    }


def task_record(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "status": "missing",
            "execution_state": None,
            "resource": None,
            "job_id": None,
            "runtime_progress": None,
            "artifact_progress": None,
            "progress_changed_at": None,
            "progress_stale_seconds": None,
            "stale_progress_labels": [],
        }
    attempts = value.get("attempts") or []
    attempt = attempts[-1] if attempts else {}
    execution_state = attempt.get("last_state")
    if execution_state is None:
        ssh_status = str(attempt.get("last_status") or "").partition(" ")[0]
        execution_state = {
            "RUNNING": "Running",
            "STARTING": "Deploying",
            "FINISHED": "Completed",
        }.get(ssh_status.upper())
    return {
        "status": value.get("status", "unknown"),
        "execution_state": execution_state,
        "resource": attempt.get("resource"),
        "job_id": attempt.get("job_id"),
        "completed_at": value.get("completed_at"),
        "failure": value.get("last_failure") or attempt.get("failure"),
        "disabled_reason": value.get("disabled_reason"),
        "runtime_progress": value.get("runtime_progress"),
        "artifact_progress": value.get("artifact_progress"),
        "progress_changed_at": value.get("runtime_progress_changed_at")
        or value.get("artifact_progress_changed_at"),
        "progress_stale_seconds": value.get("artifact_stale_seconds"),
        "stale_progress_labels": value.get("stale_progress_labels", []),
    }


def resource_metrics(snapshot: dict[str, Any]) -> dict[str, Any]:
    resources = snapshot.get("resources", {})
    beijing = resources.get("beijing", {})
    backup = beijing.get("backup", {})
    return {
        "north_primary": {
            "active_gpus": beijing.get("owned_active_gpus"),
            "queued_gpus": beijing.get("owned_queued_gpus"),
            "gpu_limit": beijing.get("personal_limit"),
        },
        "north_backup": {
            "active_gpus": backup.get("identity_active_gpus"),
            "queued_gpus": backup.get("identity_queued_gpus"),
            "gpu_limit": backup.get("personal_limit"),
            "submission_enabled": backup.get("submission_enabled"),
        },
        "east": {
            "active_gpus": resources.get("Robot-East-H20", {}).get(
                "active_gpus_all_users"
            ),
            "capacity": resources.get("Robot-East-H20", {}).get("capacity"),
        },
        "gf1": resources.get("gf1", {}),
        "local": resources.get("local", {}),
    }


def heartbeat_metrics(
    snapshot: dict[str, Any], tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    prefix = "temporal_grounding_"
    suffix = "_train"
    active_names = {
        task_id[len(prefix) : -len(suffix)]: {
            "Robot-North-H20": "beijing",
        }.get(row.get("resource"), row.get("resource"))
        for task_id, row in tasks.items()
        if task_id.startswith(f"{prefix}tg4_")
        and task_id.endswith(suffix)
        and row["status"] == "running"
        and row["execution_state"] == "Running"
    }
    status_priority = {
        "RUNNING": 5,
        "INITIALIZING": 4,
        "STARTING": 3,
        "WAITING_FOR_LOG": 2,
        "STALE_LOG": 1,
    }
    result: dict[str, Any] = {}
    for resource, values in snapshot.get("resources", {}).items():
        for name, heartbeat in values.get("watched_tasks", {}).items():
            if name not in active_names:
                continue
            active_resource = active_names[name]
            if active_resource is not None and resource != active_resource:
                continue
            candidate = {"resource": resource, **heartbeat}
            candidate["status"] = str(candidate.get("status") or "").partition(" ")[0]
            current = result.get(name)
            candidate_rank = (
                status_priority.get(candidate.get("status"), 0),
                int(candidate.get("step") or -1),
            )
            current_rank = (
                status_priority.get(current.get("status"), 0),
                int(current.get("step") or -1),
            ) if current else (-1, -1)
            if candidate_rank > current_rank:
                result[name] = candidate
    return result


def transitions(
    previous: dict[str, Any] | None, tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    if not previous:
        return []
    prior_tasks = {
        **previous.get("tasks", {}),
        **previous.get("auxiliary_tasks", {}),
    }
    changes = []
    for task_id, current in tasks.items():
        prior = prior_tasks.get(task_id, {})
        before = (
            prior.get("status"),
            prior.get("execution_state"),
            prior.get("job_id"),
        )
        after = (
            current.get("status"),
            current.get("execution_state"),
            current.get("job_id"),
        )
        if before != after:
            changes.append(
                {"task_id": task_id, "before": list(before), "after": list(after)}
            )
    return changes


def collect(
    *,
    todo_path: Path,
    snapshot_path: Path,
    state_path: Path,
    previous: dict[str, Any] | None = None,
    max_snapshot_age_seconds: int = 300,
    now: datetime | None = None,
    repo_path: Path = REPO,
) -> dict[str, Any]:
    observed_at = now or utc_now()
    errors: list[str] = []
    try:
        todo = todo_metrics(todo_path)
    except Exception as exc:
        todo = {"path": str(todo_path)}
        errors.append(f"TODO: {type(exc).__name__}: {exc}")
    try:
        snapshot = load_json(snapshot_path)
    except Exception as exc:
        snapshot = {}
        errors.append(f"snapshot: {type(exc).__name__}: {exc}")
    try:
        state = load_json(state_path)
    except Exception as exc:
        state = {}
        errors.append(f"state: {type(exc).__name__}: {exc}")

    snapshot_timestamp = snapshot.get("timestamp")
    snapshot_age = None
    if snapshot_timestamp:
        try:
            snapshot_age = max(
                0.0,
                (observed_at - parse_timestamp(snapshot_timestamp)).total_seconds(),
            )
        except Exception as exc:
            errors.append(f"snapshot timestamp: {type(exc).__name__}: {exc}")
    else:
        errors.append("snapshot timestamp: missing")
    scheduler_healthy = snapshot_age is not None and snapshot_age <= max_snapshot_age_seconds
    if not scheduler_healthy:
        errors.append(
            f"scheduler snapshot stale: age={snapshot_age!r}s, "
            f"limit={max_snapshot_age_seconds}s"
        )

    state_tasks = state.get("tasks", {})
    tasks = {task_id: task_record(state_tasks.get(task_id)) for task_id in EXPECTED_TASK_IDS}
    auxiliary_tasks = {
        task_id: task_record(state_tasks.get(task_id))
        for task_id in AUXILIARY_TASK_IDS
    }
    status_counts = dict(sorted(Counter(row["status"] for row in tasks.values()).items()))
    missing = sorted(task_id for task_id, row in tasks.items() if row["status"] == "missing")
    analyses = analysis_artifact_metrics(repo_path, state_tasks)
    for name, row in analyses["analyses"].items():
        if row["status"] == "invalid":
            errors.append(f"{name} analysis artifact: {row['error']}")
    def task_is_terminal(task_id: str, row: dict[str, Any]) -> bool:
        return row["status"] == "completed"

    incomplete = sorted(
        task_id for task_id, row in tasks.items() if not task_is_terminal(task_id, row)
    )
    complete = (
        not errors
        and not missing
        and not incomplete
        and analyses["complete"]
        and todo.get("completion_synced", False)
    )
    return {
        "timestamp": isoformat(observed_at),
        "monitor_status": "complete" if complete else ("degraded" if errors else "active"),
        "complete": complete,
        "errors": errors,
        "todo": todo,
        "scheduler": {
            "snapshot_path": str(snapshot_path),
            "state_path": str(state_path),
            "snapshot_timestamp": snapshot_timestamp,
            "snapshot_age_seconds": snapshot_age,
            "healthy": scheduler_healthy,
        },
        "resources": resource_metrics(snapshot),
        "queue_inventory": snapshot.get("queue_inventory", {}),
        "task_summary": {
            "expected": len(EXPECTED_TASK_IDS),
            "status_counts": status_counts,
            "missing": missing,
            "incomplete_count": len(incomplete),
            "incomplete": incomplete,
        },
        "tasks": tasks,
        "auxiliary_tasks": auxiliary_tasks,
        "final_analyses": analyses,
        "heartbeats": heartbeat_metrics(snapshot, tasks),
        "transitions": transitions(previous, {**tasks, **auxiliary_tasks}),
    }


def render_markdown(record: dict[str, Any]) -> str:
    summary = record["task_summary"]
    scheduler = record["scheduler"]
    lines = [
        "# Paper TODO Hourly Monitor",
        "",
        f"Updated: `{record['timestamp']}`",
        f"Status: `{record['monitor_status']}`",
        f"Scheduler healthy: `{scheduler['healthy']}` "
        f"(snapshot age `{scheduler['snapshot_age_seconds']}` seconds)",
        f"Frozen task set: `{summary['expected']}`; incomplete: "
        f"`{summary['incomplete_count']}`",
        f"TODO completion synced: `{record['todo'].get('completion_synced', False)}`",
        "",
        "## Status Counts",
        "",
        "| Status | Tasks |",
        "|---|---:|",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| `{status}` | {count} |")
    lines.extend(["", "## Training Heartbeats", ""])
    if record["heartbeats"]:
        lines.extend(
            [
                "| Task | Resource | Step | ETA (h) | Health | Status |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for name, row in sorted(record["heartbeats"].items()):
            lines.append(
                f"| `{name}` | {row.get('resource', '-')} | "
                f"{row.get('step', '-')} | {row.get('eta_hours', '-')} | "
                f"{row.get('health', '-')} | {row.get('status', '-')} |"
            )
    else:
        lines.append("No active TG4 heartbeat was reported.")
    active_evaluations = {
        task_id: row
        for task_id, row in {
            **record["tasks"],
            **record.get("auxiliary_tasks", {}),
        }.items()
        if row["status"] == "running"
        and (row.get("runtime_progress") or row.get("artifact_progress"))
    }
    lines.extend(["", "## Active Evaluations", ""])
    if active_evaluations:
        lines.extend(
            [
                "| Task | Execution | Runtime progress | Artifacts | Stale labels | Stale (s) |",
                "|---|---|---|---|---|---:|",
            ]
        )
        for task_id, row in sorted(active_evaluations.items()):
            lines.append(
                f"| `{task_id}` | {row.get('execution_state') or '-'} | "
                f"{row.get('runtime_progress') or '-'} | "
                f"{row.get('artifact_progress') or '-'} | "
                f"{','.join(row.get('stale_progress_labels', [])) or '-'} | "
                f"{row.get('progress_stale_seconds', '-')} |"
            )
    else:
        lines.append("No active evaluation progress was reported.")
    lines.extend(
        [
            "",
            "## Final Analyses",
            "",
            "| Analysis | Task | Status | Report | Marker |",
            "|---|---|---|---|---|",
        ]
    )
    for name, row in sorted(record["final_analyses"]["analyses"].items()):
        lines.append(
            f"| `{name}` | `{row['task']['status']}` | `{row['status']}` | "
            f"`{row['present']['report']}` | "
            f"`{row['present']['marker']}` |"
        )
    lines.extend(["", "## Transitions", ""])
    if record["transitions"]:
        for change in record["transitions"]:
            lines.append(
                f"- `{change['task_id']}`: `{change['before']}` -> "
                f"`{change['after']}`"
            )
    else:
        lines.append("No task-state transition since the previous hourly poll.")
    if record["errors"]:
        lines.extend(["", "## Alerts", ""])
        lines.extend(f"- {error}" for error in record["errors"])
    lines.append("")
    return "\n".join(lines)


def persist(
    record: dict[str, Any], *, jsonl_path: Path, latest_json: Path, latest_md: Path
) -> None:
    atomic_write(latest_json, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    atomic_write(latest_md, render_markdown(record))
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_previous(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=3600)
    parser.add_argument("--max-snapshot-age-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-polls", type=int)
    parser.add_argument("--todo", type=Path, default=DEFAULT_TODO)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--latest-json", type=Path, default=DEFAULT_LATEST_JSON)
    parser.add_argument("--latest-md", type=Path, default=DEFAULT_LATEST_MD)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--repo", type=Path, default=REPO)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    if args.max_snapshot_age_seconds <= 0:
        raise SystemExit("--max-snapshot-age-seconds must be positive")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"monitor already running: {args.lock}", flush=True)
            return 2
        monitor_started = utc_now()
        write_lock_status(
            lock_handle, started=monitor_started, last_poll=None, next_poll=None
        )
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        polls = 0
        while not STOP_REQUESTED:
            previous = load_previous(args.latest_json)
            record = collect(
                todo_path=args.todo,
                snapshot_path=args.snapshot,
                state_path=args.state,
                previous=previous,
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
                repo_path=args.repo,
            )
            persist(
                record,
                jsonl_path=args.jsonl,
                latest_json=args.latest_json,
                latest_md=args.latest_md,
            )
            polls += 1
            print(
                f"[{record['timestamp']}] status={record['monitor_status']} "
                f"incomplete={record['task_summary']['incomplete_count']} "
                f"transitions={len(record['transitions'])}",
                flush=True,
            )
            if record["complete"] or args.once:
                return 0
            if args.max_polls is not None and polls >= args.max_polls:
                return 0
            next_poll = next_interval_boundary(utc_now(), args.interval_seconds)
            write_lock_status(
                lock_handle,
                started=monitor_started,
                last_poll=parse_timestamp(record["timestamp"]),
                next_poll=next_poll,
            )
            print(f"next_poll={isoformat(next_poll)}", flush=True)
            deadline = time.monotonic() + max(
                0.0, next_poll.timestamp() - time.time()
            )
            while not STOP_REQUESTED and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

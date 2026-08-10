#!/usr/bin/env python3
"""Persist an hourly, read-only audit of the active paper GPU TODO.

Experiment execution remains owned by resource_aware_scheduler.py. This
monitor only reads its state/snapshot, records progress, and exits when the
frozen TG1A/TG2R task set is completely finished.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
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

STOP_REQUESTED = False


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def expected_task_ids() -> tuple[str, ...]:
    tasks = [
        f"temporal_grounding_tg1a_{condition}_eval"
        for condition in ("normal", "null", "persistence", "shuffled")
    ]
    tasks.append("temporal_grounding_tg2r_north_stage")
    for arm in ("future_off", "fixed_endpoint", "raw_milestone"):
        for seed in (1000, 1001, 1002):
            parent = f"temporal_grounding_tg2r_{arm}_seed{seed}_train"
            tasks.extend((parent, f"{parent}_materialize_north"))
            tasks.append(f"temporal_grounding_tg2r_{arm}_seed{seed}_eval")
    tasks.append("temporal_grounding_tg2r_training_integrity")
    return tuple(tasks)


EXPECTED_TASK_IDS = expected_task_ids()


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
    return {
        "path": str(path),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "unchecked_current_override": current.count("- [ ]"),
        "unchecked_total": text.count("- [ ]"),
        "checked_total": text.count("- [x]") + text.count("- [X]"),
    }


def task_record(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"status": "missing", "execution_state": None, "job_id": None}
    attempts = value.get("attempts") or []
    attempt = attempts[-1] if attempts else {}
    return {
        "status": value.get("status", "unknown"),
        "execution_state": attempt.get("last_state"),
        "job_id": attempt.get("job_id"),
        "completed_at": value.get("completed_at"),
        "failure": value.get("last_failure") or attempt.get("failure"),
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
        "local": resources.get("local", {}),
    }


def heartbeat_metrics(
    snapshot: dict[str, Any], tasks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    prefix = "temporal_grounding_"
    suffix = "_train"
    active_names = {
        task_id[len(prefix) : -len(suffix)]
        for task_id, row in tasks.items()
        if task_id.startswith(f"{prefix}tg2r_")
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
            candidate = {"resource": resource, **heartbeat}
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
    prior_tasks = previous.get("tasks", {})
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
    status_counts = dict(sorted(Counter(row["status"] for row in tasks.values()).items()))
    missing = sorted(task_id for task_id, row in tasks.items() if row["status"] == "missing")
    incomplete = sorted(
        task_id for task_id, row in tasks.items() if row["status"] != "completed"
    )
    complete = not errors and not missing and not incomplete
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
        "heartbeats": heartbeat_metrics(snapshot, tasks),
        "transitions": transitions(previous, tasks),
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
        lines.append("No active TG2R heartbeat was reported.")
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
        lock_handle.write(f"pid={os.getpid()} started={isoformat(utc_now())}\n")
        lock_handle.flush()
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        polls = 0
        while not STOP_REQUESTED:
            started = time.monotonic()
            previous = load_previous(args.latest_json)
            record = collect(
                todo_path=args.todo,
                snapshot_path=args.snapshot,
                state_path=args.state,
                previous=previous,
                max_snapshot_age_seconds=args.max_snapshot_age_seconds,
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
            remaining = max(0.0, args.interval_seconds - (time.monotonic() - started))
            deadline = time.monotonic() + remaining
            while not STOP_REQUESTED and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

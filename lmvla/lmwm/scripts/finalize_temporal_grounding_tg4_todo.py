#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROTOCOL = "temporal_grounding_tg4_source_decomposition_analysis_v1"
COMPARISONS = (
    "pretraining",
    "auxiliary_shaping",
    "conditioning_without_auxiliary",
    "full_total",
    "full_vs_historical_off",
    "route_interaction",
    "content_use",
)
LABEL_REPLACEMENTS = {
    "TG4-T01--T18": "TG4-T01--T18 [COMPLETE; 18/18 COMPLETE]",
    "TG4-I1": "TG4-I1 [COMPLETE]",
    "TG4-E1": "TG4-E1 [COMPLETE; 21/21 COMPLETE]",
    "TG4-A1": "TG4-A1 [COMPLETE]",
}
SUMMARY_START = "<!-- TG4_FINAL_RESULT_START -->"
SUMMARY_END = "<!-- TG4_FINAL_RESULT_END -->"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.chmod(0o664)
    os.replace(temporary, path)


def validate(report: dict[str, Any], marker_text: str) -> dict[str, dict[str, Any]]:
    if report.get("protocol") != PROTOCOL or report.get("complete") is not True:
        raise ValueError("TG4 report is not a complete frozen analysis")
    if report.get("holm_family") != list(COMPARISONS):
        raise ValueError("TG4 Holm family is incomplete or reordered")
    rows = report.get("comparisons")
    if not isinstance(rows, dict) or set(rows) != set(COMPARISONS):
        raise ValueError("TG4 report does not contain exactly seven comparisons")
    marker_lines = {line.strip() for line in marker_text.splitlines() if line.strip()}
    required = {"validated=true", f"protocol={PROTOCOL}"}
    if not required.issubset(marker_lines):
        raise ValueError("TG4 decision marker is not validated for the frozen protocol")
    for name in COMPARISONS:
        row = rows[name]
        accepted = row.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError(f"TG4 comparison {name} lacks a boolean verdict")
        if f"{name}={str(accepted).lower()}" not in marker_lines:
            raise ValueError(f"TG4 marker verdict mismatch for {name}")
        ci = row.get("hierarchical_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"TG4 comparison {name} lacks a 95% interval")
        for field in (
            "mean_effect",
            "holm_adjusted_p",
            "minimum_training_seed_task_effect",
        ):
            if not isinstance(row.get(field), (int, float)):
                raise ValueError(f"TG4 comparison {name} lacks numeric {field}")
        if not isinstance(row.get("task_safety_passed"), bool):
            raise ValueError(f"TG4 comparison {name} lacks a task-safety verdict")
    return rows


def result_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# TG4 Source-Decomposition Result",
        "",
        f"Protocol: `{PROTOCOL}`",
        "",
        "| Contrast | Mean effect | Hierarchical 95% CI | Holm p | Minimum cell | Safety | Accepted |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name in COMPARISONS:
        row = rows[name]
        lower, upper = row["hierarchical_ci95"]
        lines.append(
            f"| `{name}` | {row['mean_effect']:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] | {row['holm_adjusted_p']:.6g} | "
            f"{row['minimum_training_seed_task_effect']:+.4f} | "
            f"{str(row['task_safety_passed'])} | {str(row['accepted'])} |"
        )
    lines.extend(
        [
            "",
            "An accepted contrast satisfies all frozen statistical and task-safety gates. "
            "A rejected contrast remains reportable and does not authorize follow-up tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def update_todo(text: str, rows: dict[str, dict[str, Any]], timestamp: str) -> str:
    updated = re.sub(
        r"^Updated: .* UTC$", f"Updated: {timestamp} UTC", text, count=1, flags=re.MULTILINE
    )
    if updated == text:
        raise ValueError("TODO Updated timestamp was not found")
    for label, replacement in LABEL_REPLACEMENTS.items():
        pattern = rf"^- \[[ xX]\] \*\*{re.escape(label)}(?:\s*\[[^\n]*?\])?"
        updated, count = re.subn(
            pattern,
            f"- [x] **{replacement}",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"expected exactly one TODO gate for {label}, found {count}")
    accepted = [name for name in COMPARISONS if rows[name]["accepted"]]
    rejected = [name for name in COMPARISONS if not rows[name]["accepted"]]
    block = "\n".join(
        [
            SUMMARY_START,
            "### Final TG4 decision",
            "",
            "All 18 training cells, the integrity gate, and all 21 frozen evaluation "
            "panels completed. The prespecified analysis accepted "
            f"{', '.join(f'`{name}`' for name in accepted) if accepted else 'no contrasts'} "
            "and rejected "
            f"{', '.join(f'`{name}`' for name in rejected) if rejected else 'no contrasts'}. "
            "The canonical numerical table is in "
            "`RESULTS_temporal_grounding_tg4.md`; rejected gates do not authorize tuning.",
            SUMMARY_END,
        ]
    )
    if SUMMARY_START in updated or SUMMARY_END in updated:
        pattern = re.escape(SUMMARY_START) + r".*?" + re.escape(SUMMARY_END)
        updated, count = re.subn(pattern, block, updated, count=1, flags=re.DOTALL)
        if count != 1:
            raise ValueError("malformed existing TG4 final-result block")
    else:
        anchor = "\n### TG4 claim gates\n"
        if anchor not in updated:
            raise ValueError("TG4 claim-gate anchor was not found")
        updated = updated.replace(anchor, f"\n{block}\n{anchor}", 1)
    return updated


def finalize(
    *, report_path: Path, analysis_marker: Path, todo_path: Path, summary_path: Path,
    completion_marker: Path, lock_path: Path, now: datetime | None = None,
) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = validate(report, analysis_marker.read_text(encoding="utf-8"))
        timestamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
        todo = update_todo(todo_path.read_text(encoding="utf-8"), rows, timestamp)
        atomic_write(summary_path, result_table(rows))
        atomic_write(todo_path, todo)
        marker = "\n".join(
            [
                "validated=true",
                "protocol=temporal_grounding_tg4_todo_finalization_v1",
                f"report_sha256={sha256(report_path)}",
                f"analysis_marker_sha256={sha256(analysis_marker)}",
                f"summary_sha256={sha256(summary_path)}",
                f"todo_sha256={sha256(todo_path)}",
                "",
            ]
        )
        atomic_write(completion_marker, marker)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--analysis-marker", type=Path, required=True)
    parser.add_argument("--todo", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--completion-marker", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    finalize(
        report_path=args.report,
        analysis_marker=args.analysis_marker,
        todo_path=args.todo,
        summary_path=args.summary,
        completion_marker=args.completion_marker,
        lock_path=args.lock,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("normal", "null", "persistence")
SCHEMA_ERROR = "contains unsupported keys ['temporal_grounding_context']"


def archive_failed_runtime_v4(
    repo: Path, condition: str, *, check_only: bool = False
) -> Path | None:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported TG1A retry condition: {condition}")

    result_root = (
        repo
        / "lmvla/lawam/results/eval_runs/robotwin"
        / f"temporal_grounding_tg1a_{condition}"
    )
    archive = result_root.with_name(result_root.name + ".failed_runtime_v4_schema")
    if not result_root.exists():
        return None
    if archive.exists():
        raise FileExistsError(f"TG1A v4 archive already exists: {archive}")

    summaries = list(result_root.glob("seed*/**/summary.json"))
    if summaries:
        raise RuntimeError(
            f"refusing to archive TG1A {condition}: found {len(summaries)} summaries"
        )

    statuses = list(result_root.glob("seed*/**/.task_status/*.json"))
    if len(statuses) != 24:
        raise RuntimeError(
            f"refusing to archive TG1A {condition}: expected 24 task statuses, "
            f"found {len(statuses)}"
        )
    status_values = [json.loads(path.read_text()).get("status") for path in statuses]
    if any(status != "failed" for status in status_values):
        raise RuntimeError(
            f"refusing to archive TG1A {condition}: not every task status is failed"
        )

    server_logs = list(result_root.glob("seed*/**/workers/*/server.log"))
    matching_logs = [
        path for path in server_logs if SCHEMA_ERROR in path.read_text(errors="replace")
    ]
    if len(server_logs) != 4 or len(matching_logs) != 4:
        raise RuntimeError(
            f"refusing to archive TG1A {condition}: expected the schema error in "
            f"all 4 server logs, found logs={len(server_logs)} matches={len(matching_logs)}"
        )

    if check_only:
        return archive

    result_root.rename(archive)

    if condition == "normal":
        feature_root = (
            repo / "logs/temporal_grounding/tg1a/predicted_endpoint_features"
        )
        feature_archive = feature_root.with_name(
            feature_root.name + ".failed_runtime_v4_schema"
        )
        capture_marker = (
            repo / "logs/temporal_grounding/tg1a/normal_capture_complete.json"
        )
        if capture_marker.exists():
            raise RuntimeError(
                f"refusing to archive normal feature root: capture marker exists: {capture_marker}"
            )
        if feature_archive.exists():
            raise FileExistsError(
                f"TG1A v4 feature archive already exists: {feature_archive}"
            )
        if feature_root.exists():
            if any(feature_root.iterdir()):
                raise RuntimeError(
                    f"refusing to archive non-empty normal feature root: {feature_root}"
                )
            feature_root.rename(feature_archive)

    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    archive = archive_failed_runtime_v4(
        args.repo.resolve(), args.condition, check_only=args.check_only
    )
    status = "clean"
    if archive:
        status = "validated" if args.check_only else "archived"
    print(
        json.dumps(
            {
                "condition": args.condition,
                "status": status,
                "archive": str(archive) if archive else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

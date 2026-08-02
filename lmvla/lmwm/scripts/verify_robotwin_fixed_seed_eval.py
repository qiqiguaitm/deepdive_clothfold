#!/usr/bin/env python3
"""Verify that evaluation summaries exactly follow a frozen scene-seed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Verify the observed cells without requiring the full manifest coverage.",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        (int(eval_seed), task): [int(value) for value in seeds]
        for eval_seed, task_map in payload["eval_seeds"].items()
        for task, seeds in task_map.items()
    }
    observed: dict[tuple[int, str], tuple[list[int], Path]] = {}
    expected_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    for root in args.root:
        for summary_path in sorted(root.rglob("summary.json")):
            match = re.search(r"/seed(\d+)(?:/|$)", str(summary_path))
            if not match:
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            key = (int(match.group(1)), str(summary.get("task_name")))
            if key not in expected:
                continue
            seeds = [int(item["seed"]) for item in summary.get("episodes", [])]
            manifest_meta = summary.get("fixed_seed_manifest") or {}
            if manifest_meta.get("sha256") != expected_sha:
                raise SystemExit(f"{summary_path}: fixed-seed manifest SHA mismatch")
            if key in observed:
                raise SystemExit(f"duplicate result cell {key}: {observed[key][1]} and {summary_path}")
            observed[key] = (seeds, summary_path)

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if extra or (missing and not args.allow_partial):
        raise SystemExit(f"result cell mismatch: missing={missing}, extra={extra}")
    if not observed:
        raise SystemExit("result cell mismatch: no manifest cells were observed")
    for key, (actual_seeds, summary_path) in observed.items():
        expected_seeds = expected[key]
        if actual_seeds != expected_seeds:
            raise SystemExit(
                f"{summary_path}: scene seeds differ for {key}; "
                f"expected={expected_seeds}, actual={actual_seeds}"
            )

    print(
        json.dumps(
            {
                "cells": len(observed),
                "complete": not missing,
                "manifest_sha256": expected_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

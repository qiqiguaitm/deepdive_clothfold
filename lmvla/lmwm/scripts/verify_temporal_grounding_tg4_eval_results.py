#!/usr/bin/env python3
"""Strictly validate complete TG4 fixed-scene evaluation summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = {
        (int(eval_seed), task): [int(value) for value in seeds]
        for eval_seed, task_map in payload["eval_seeds"].items()
        for task, seeds in task_map.items()
    }
    expected_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    observed: dict[tuple[int, str], tuple[list[int], Path]] = {}

    for summary_path in sorted(args.root.rglob("summary.json")):
        match = re.search(r"/seed(\d+)(?:/|$)", str(summary_path))
        if not match:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (int(match.group(1)), str(summary.get("task_name")))
        if key not in expected:
            continue
        fixed = summary.get("fixed_seed_manifest")
        if not isinstance(fixed, dict) or fixed.get("sha256") != expected_sha:
            raise SystemExit(f"{summary_path}: fixed-seed manifest SHA mismatch")
        if fixed.get("eval_seed") != key[0]:
            raise SystemExit(f"{summary_path}: fixed-seed eval_seed mismatch")
        episodes = summary.get("episodes")
        if not isinstance(episodes, list):
            raise SystemExit(f"{summary_path}: episodes must be a list")
        if summary.get("n_episodes") != len(episodes):
            raise SystemExit(f"{summary_path}: n_episodes does not match episodes")
        seeds = []
        for index, item in enumerate(episodes):
            if not isinstance(item, dict) or type(item.get("seed")) is not int:
                raise SystemExit(f"{summary_path}: episode {index} has invalid seed")
            if type(item.get("success")) is not bool:
                raise SystemExit(f"{summary_path}: episode {index} has invalid success")
            seeds.append(item["seed"])
        if key in observed:
            raise SystemExit(f"duplicate result cell {key}: {observed[key][1]} and {summary_path}")
        observed[key] = (seeds, summary_path)

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise SystemExit(f"result cell mismatch: missing={missing}")
    for key, (actual_seeds, summary_path) in observed.items():
        expected_seeds = expected[key]
        if len(set(expected_seeds)) != len(expected_seeds):
            raise SystemExit(f"manifest contains duplicate scene seeds for {key}")
        if len(set(actual_seeds)) != len(actual_seeds):
            raise SystemExit(f"{summary_path}: result contains duplicate scene seeds for {key}")
        if len(actual_seeds) != len(expected_seeds) or set(actual_seeds) != set(expected_seeds):
            raise SystemExit(
                f"{summary_path}: scene seeds differ for {key}; "
                f"expected={expected_seeds}, actual={actual_seeds}"
            )

    print(json.dumps({"cells": len(observed), "complete": True, "manifest_sha256": expected_sha}, sort_keys=True))


if __name__ == "__main__":
    main()

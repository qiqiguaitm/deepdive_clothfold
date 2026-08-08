#!/usr/bin/env python3
"""Read-only audit for the proposed TG1 fixed-scene retry amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = (
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
)
EVAL_SEEDS = (0, 1, 2, 3)
PANELS = {
    "tg1a": (
        "temporal_grounding_tg1a_normal",
        "temporal_grounding_tg1a_shuffled",
        "temporal_grounding_tg1a_null",
        "temporal_grounding_tg1a_persistence",
    ),
    "tg1b": (
        "temporal_grounding_tg1b_future_off_e36",
        "temporal_grounding_tg1b_future_off_e50",
        "temporal_grounding_tg1b_local_wm_e36",
        "temporal_grounding_tg1b_local_wm_e50",
    ),
}
CAP_PATTERN = re.compile(r"^export ROBOTWIN_FIXED_SEED_MAX_ATTEMPTS=(\d+)$", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_retry_cap(path: Path) -> int:
    matches = CAP_PATTERN.findall(path.read_text())
    if len(matches) != 1:
        raise ValueError(f"expected one fixed retry cap in {path}, found {matches}")
    return int(matches[0])


def expected_cells() -> set[tuple[int, str]]:
    return {(seed, task) for seed in EVAL_SEEDS for task in TASKS}


def summary_cell(path: Path) -> tuple[int, str]:
    seed_matches = [part for part in path.parts if re.fullmatch(r"seed[0-3]", part)]
    if len(seed_matches) != 1:
        raise ValueError(f"cannot identify one eval seed from {path}")
    return int(seed_matches[0][4:]), path.parent.name


def validate_summary(
    path: Path, scene_manifest: dict[str, Any], scene_sha256: str
) -> tuple[int, str]:
    seed, task = summary_cell(path)
    if task not in scene_manifest["eval_seeds"][str(seed)]:
        raise ValueError(f"unexpected task {task} in {path}")
    payload = json.loads(path.read_text())
    provenance = payload.get("fixed_seed_manifest", {})
    episodes = payload.get("episodes", [])
    expected = scene_manifest["eval_seeds"][str(seed)][task]
    observed = [episode.get("seed") for episode in episodes]
    checks = {
        "task_name": payload.get("task_name") == task,
        "n_episodes": payload.get("n_episodes") == len(expected) == 50,
        "manifest_sha256": provenance.get("sha256") == scene_sha256,
        "manifest_eval_seed": provenance.get("eval_seed") == seed,
        "manifest_task": provenance.get("task_name") == task,
        "manifest_count": provenance.get("count") == len(expected),
        "exact_ordered_scene_ids": observed == expected,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"invalid frozen summary {path}: {', '.join(failed)}")
    return seed, task


def task_status_cells(root: Path) -> dict[tuple[int, str], str]:
    records: dict[tuple[int, str], str] = {}
    for path in root.glob("seed*/**/.task_status/*.json"):
        seed_matches = [part for part in path.parts if re.fullmatch(r"seed[0-3]", part)]
        if len(seed_matches) != 1:
            continue
        cell = (int(seed_matches[0][4:]), path.stem)
        status = str(json.loads(path.read_text()).get("status", "unknown"))
        records[cell] = status
    return records


def audit_root(root: Path, scene_manifest: dict[str, Any], scene_sha256: str) -> dict[str, Any]:
    expected = expected_cells()
    if not root.exists():
        return {
            "exists": False,
            "valid_summaries": 0,
            "failed_cells": [],
            "missing_cells": len(expected),
            "complete": False,
        }

    summaries: dict[tuple[int, str], Path] = {}
    for path in root.glob("seed*/**/tasks/*/summary.json"):
        cell = validate_summary(path, scene_manifest, scene_sha256)
        if cell in summaries:
            raise ValueError(f"duplicate summary cell {cell} in {root}")
        summaries[cell] = path
    statuses = task_status_cells(root)
    failed = sorted(cell for cell, status in statuses.items() if status == "failed")
    missing = expected - summaries.keys()
    return {
        "exists": True,
        "valid_summaries": len(summaries),
        "failed_cells": [
            {"eval_seed": seed, "task": task} for seed, task in failed
        ],
        "missing_cells": len(missing),
        "complete": set(summaries) == expected and not failed,
    }


def build_audit(repo: Path) -> dict[str, Any]:
    scene_path = repo / "lmvla/lmwm/data/robotwin_pi05_confirmatory_scene_seeds_v1.json"
    scene_manifest = json.loads(scene_path.read_text())
    if set(scene_manifest.get("tasks", [])) != set(TASKS):
        raise ValueError("scene manifest tasks differ from the frozen panel")
    if sorted(map(int, scene_manifest.get("eval_seeds", {}))) != list(EVAL_SEEDS):
        raise ValueError("scene manifest eval seeds differ from the frozen panel")
    if scene_manifest.get("episodes_per_cell") != 50:
        raise ValueError("scene manifest does not contain 50 episodes per cell")

    runners = {
        "tg1a": repo / "train_scripts/kai/eval/run_temporal_grounding_tg1a_formal.sh",
        "tg1b": repo / "train_scripts/kai/eval/run_temporal_grounding_tg1b_formal.sh",
    }
    current_caps = {name: read_retry_cap(path) for name, path in runners.items()}
    if set(current_caps.values()) != {3}:
        raise ValueError(f"frozen runners no longer share retry cap 3: {current_caps}")

    result_base = repo / "lmvla/lawam/results/eval_runs/robotwin"
    roots = {
        panel: {
            name: audit_root(result_base / name, scene_manifest, sha256(scene_path))
            for name in names
        }
        for panel, names in PANELS.items()
    }
    valid_summary_count = sum(
        record["valid_summaries"]
        for panel in roots.values()
        for record in panel.values()
    )
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Read-only TG1A/TG1B fixed-scene retry amendment audit",
        "authorization": {
            "status": "audit_only",
            "activated": False,
            "jobs_modified": False,
            "scheduler_readiness_marker": False,
            "explicit_operator_authorization_required": True,
        },
        "frozen_scene_manifest": {
            "path": str(scene_path.relative_to(repo)),
            "sha256": sha256(scene_path),
            "tasks": list(TASKS),
            "eval_seeds": list(EVAL_SEEDS),
            "episodes_per_cell": 50,
            "cells_per_condition": 24,
            "accepted_episodes_per_condition": 1200,
        },
        "current_runner_retry_caps": current_caps,
        "runner_sha256": {name: sha256(path) for name, path in runners.items()},
        "proposed_common_retry_cap": 500,
        "required_rerun_conditions": {
            panel: list(names) for panel, names in PANELS.items()
        },
        "existing_result_roots": roots,
        "validation": {
            "all_existing_summaries_match_frozen_scene_manifest": True,
            "existing_valid_summary_count": valid_summary_count,
        },
        "activation_invariants": [
            "Archive every existing incomplete root before relaunch.",
            "Rerun all four conditions in each affected panel under one common cap.",
            "Do not reuse a partial summary, replace a scene, or drop a failed cell.",
            "Require exactly 24 valid cells and 1,200 frozen scene identities per condition.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(build_audit(args.repo.resolve()), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the six-task train-reference and P0-heldout panel for CRAVE R0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TASK_BLOCKS = {
    "beat_block_hammer": 1,
    "blocks_ranking_rgb": 2,
    "blocks_ranking_size": 3,
    "handover_block": 8,
    "stack_blocks_three": 44,
    "stack_blocks_two": 45,
}
EPISODES_PER_TASK = 550
REFERENCE_EPISODES_PER_TASK = 200
SELECTION_SEED = 20260804


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_reference_selection(episodes: list[int], task: str, count: int) -> list[int]:
    if len(episodes) < count:
        raise ValueError(f"{task}: only {len(episodes)} train episodes for {count} references")
    return sorted(
        episodes,
        key=lambda episode: hashlib.sha256(
            f"{SELECTION_SEED}:{task}:{episode}".encode()
        ).digest(),
    )[:count]


def build_selection(split: dict, panel: dict[str, np.ndarray], feature_dir: Path) -> tuple[dict, np.ndarray, list[int]]:
    heldout = set(map(int, split["heldout_episodes"]))
    train = set(map(int, split["train_episodes"]))
    panel_episodes = panel["cur_ep"].astype(np.int64)
    selected_rows = []
    tasks = {}
    required = set()
    for task, block in TASK_BLOCKS.items():
        lower = block * EPISODES_PER_TASK
        upper = lower + EPISODES_PER_TASK
        task_heldout = sorted(episode for episode in heldout if lower <= episode < upper)
        task_train = sorted(episode for episode in train if lower <= episode < upper)
        references = stable_reference_selection(
            task_train, task, REFERENCE_EPISODES_PER_TASK
        )
        row_indices = np.flatnonzero(
            (panel_episodes >= lower) & (panel_episodes < upper)
        )
        row_episode_set = set(map(int, panel_episodes[row_indices]))
        if row_episode_set != set(task_heldout):
            raise ValueError(f"{task}: panel does not cover every heldout episode exactly as expected")
        if set(references) & set(task_heldout):
            raise ValueError(f"{task}: reference/heldout leakage")
        selected_rows.extend(map(int, row_indices))
        required.update(references)
        required.update(task_heldout)
        tasks[task] = {
            "official_block": block,
            "episode_range": [lower, upper - 1],
            "reference_episodes": references,
            "reference_count": len(references),
            "heldout_episodes": task_heldout,
            "heldout_episode_count": len(task_heldout),
            "panel_row_count": len(row_indices),
        }

    selected_rows_array = np.asarray(sorted(selected_rows), dtype=np.int64)
    required_list = sorted(required)
    available = {
        int(path.stem[2:])
        for path in feature_dir.glob("ep*.npz")
        if path.stem[2:].isdigit()
    }
    missing = sorted(set(required_list) - available)
    manifest = {
        "schema_version": 1,
        "protocol": "pi05_crave_r0_six_task_selection_v1",
        "selection_seed": SELECTION_SEED,
        "reference_selection": "first 200 episodes after SHA-256 ordering of seed:task:episode",
        "episode_split_unit": "episode",
        "tasks": tasks,
        "task_count": len(tasks),
        "reference_episode_count": sum(row["reference_count"] for row in tasks.values()),
        "heldout_episode_count": sum(row["heldout_episode_count"] for row in tasks.values()),
        "panel_row_count": len(selected_rows_array),
        "required_episode_count": len(required_list),
        "feature_coverage_at_build": {
            "available_required_episodes": len(set(required_list) & available),
            "missing_required_episodes": missing,
            "missing_required_episode_count": len(missing),
        },
    }
    return manifest, selected_rows_array, required_list


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--heldout-panel", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.gate.name not in {"p0_gate.accepted", "p0_gate.rejected"}:
        raise ValueError("R0 selection requires a frozen accepted or rejected P0 verdict")
    for path in (args.gate, args.split, args.heldout_panel):
        if not path.is_file():
            raise FileNotFoundError(path)
    panel_npz = np.load(args.heldout_panel)
    panel = {key: panel_npz[key] for key in panel_npz.files}
    manifest, selected_rows, required = build_selection(
        json.loads(args.split.read_text()), panel, args.feature_dir
    )
    manifest.update(
        {
            "p0_verdict": args.gate.name.removeprefix("p0_gate."),
            "p0_gate_sha256": sha256(args.gate),
            "episode_split_sha256": sha256(args.split),
            "source_heldout_panel_sha256": sha256(args.heldout_panel),
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "panel.npz",
        **{key: value[selected_rows] for key, value in panel.items()},
    )
    atomic_text(
        args.output / "required_episodes.txt",
        "".join(f"{episode}\n" for episode in required),
    )
    atomic_text(
        args.output / "selection_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    manifest["panel_sha256"] = sha256(args.output / "panel.npz")
    manifest["required_episodes_sha256"] = sha256(
        args.output / "required_episodes.txt"
    )
    atomic_text(
        args.output / "selection_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(args.output / "READY_SELECTION", "ready\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

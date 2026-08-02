#!/usr/bin/env python3
"""Build compact task-specific current-to-ground-truth-milestone retrieval banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TASK_NAMES = {
    0: "beat_block_hammer",
    1: "stack_blocks_two",
    2: "stack_blocks_three",
    3: "blocks_ranking_rgb",
    4: "blocks_ranking_size",
    5: "handover_block",
}


def choose_episode_stratified(
    indices: np.ndarray,
    cur_ep: np.ndarray,
    entries_per_task: int,
    rng: np.random.Generator,
) -> np.ndarray:
    episodes = np.unique(cur_ep[indices])
    rng.shuffle(episodes)
    selected: list[int] = []
    for episode in episodes:
        episode_indices = indices[cur_ep[indices] == episode]
        selected.append(int(rng.choice(episode_indices)))
        if len(selected) == entries_per_task:
            break
    if len(selected) < entries_per_task:
        remaining = rng.choice(
            indices,
            size=entries_per_task - len(selected),
            replace=len(indices) < entries_per_task - len(selected),
        )
        selected.extend(map(int, remaining))
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entries-per-task", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pairs = np.load(args.pairs)
    arrays = {name: pairs[name] for name in pairs.files}
    required = {"cur_ep", "cur_fi", "tgt_fi", "pair_task"}
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"pairs file is missing arrays: {sorted(missing)}")
    if args.entries_per_task < 1:
        raise ValueError("entries-per-task must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest: dict[str, object] = {
        "definition": (
            "state-dependent demo retrieval: nearest pooled current grid, "
            "inject its paired ground-truth milestone grid"
        ),
        "entries_per_task": args.entries_per_task,
        "seed": args.seed,
        "tasks": {},
    }

    for task_id, task_name in TASK_NAMES.items():
        candidates = np.flatnonzero(arrays["pair_task"] == task_id)
        chosen = choose_episode_stratified(
            candidates, arrays["cur_ep"], args.entries_per_task, rng
        )
        current = []
        target = []
        for index in chosen:
            episode = int(arrays["cur_ep"][index])
            grid = np.load(args.feat / f"ep{episode}.npz")["grid"]
            current.append(grid[int(arrays["cur_fi"][index])])
            target.append(grid[int(arrays["tgt_fi"][index])])
        current_grid = np.asarray(current, dtype=np.float16)
        target_grid = np.asarray(target, dtype=np.float16)
        current_pooled = current_grid.astype(np.float32).mean(axis=1)
        current_pooled /= np.linalg.norm(current_pooled, axis=1, keepdims=True) + 1e-8
        output = args.output / f"{task_name}.npz"
        np.savez_compressed(
            output,
            current_pooled=current_pooled.astype(np.float32),
            target_grid=target_grid,
            cur_ep=arrays["cur_ep"][chosen].astype(np.int64),
            cur_fi=arrays["cur_fi"][chosen].astype(np.int64),
            tgt_fi=arrays["tgt_fi"][chosen].astype(np.int64),
        )
        manifest["tasks"][task_name] = {
            "entries": int(len(chosen)),
            "unique_episodes": int(len(np.unique(arrays["cur_ep"][chosen]))),
            "path": str(output),
        }
        print(f"{task_name}: {len(chosen)} entries -> {output}", flush=True)

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

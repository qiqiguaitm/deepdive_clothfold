#!/usr/bin/env python3
"""Freeze the Stack-3/Hammer episode split for the pi0.5 spatial S0 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TASKS = {"hammer": 0, "stack_three": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-per-task", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    pairs_path = Path(args.pairs).resolve()
    pairs = np.load(pairs_path)
    rng = np.random.default_rng(args.seed)
    task_splits: dict[str, dict[str, list[int]]] = {}
    for task, task_id in TASKS.items():
        episodes = np.unique(pairs["cur_ep"][pairs["pair_task"] == task_id]).astype(int)
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        heldout = np.sort(shuffled[: args.heldout_per_task])
        train = np.sort(shuffled[args.heldout_per_task :])
        task_splits[task] = {"train": train.tolist(), "heldout": heldout.tolist()}

    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "pairs": str(pairs_path),
        "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "heldout_per_task": args.heldout_per_task,
        "tasks": task_splits,
        "train_episodes": sorted(sum((item["train"] for item in task_splits.values()), [])),
        "heldout_episodes": sorted(sum((item["heldout"] for item in task_splits.values()), [])),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: len(payload[key]) for key in ("train_episodes", "heldout_episodes")}, indent=2))


if __name__ == "__main__":
    main()

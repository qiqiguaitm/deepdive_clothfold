#!/usr/bin/env python3
"""Freeze exact held-out frames and RNG keys for the pi0.5 spatial S0 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--episodes-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=20260801)
    args = parser.parse_args()

    split_path = Path(args.split).resolve()
    pairs_path = Path(args.pairs).resolve()
    episodes_path = Path(args.episodes_jsonl).resolve()
    split = json.loads(split_path.read_text())
    lengths = {
        int(row["episode_index"]): int(row["length"])
        for row in map(json.loads, episodes_path.read_text().splitlines())
    }
    task_by_episode = {
        int(episode): task
        for task, task_split in split["tasks"].items()
        for episode in task_split["heldout"]
    }

    pairs = np.load(pairs_path)
    samples: list[dict[str, int | str]] = []
    for episode in split["heldout_episodes"]:
        episode = int(episode)
        mask = pairs["cur_ep"] == episode
        cur = pairs["cur_fi"][mask].astype(int)
        tgt = pairs["tgt_fi"][mask].astype(int)
        valid = cur <= lengths[episode] - args.action_horizon
        cur = cur[valid]
        tgt = tgt[valid]
        if len(cur) < args.samples_per_episode:
            raise ValueError(f"episode {episode} has only {len(cur)} valid paired frames")
        positions = np.linspace(0, len(cur) - 1, args.samples_per_episode + 2)[1:-1]
        indices = np.rint(positions).astype(int)
        if len(np.unique(indices)) != args.samples_per_episode:
            raise ValueError(f"episode {episode} produced duplicate frame selections")
        for index in indices:
            samples.append(
                {
                    "task": task_by_episode[episode],
                    "episode_index": episode,
                    "frame_index": int(cur[index]),
                    "target_frame_index": int(tgt[index]),
                }
            )

    by_task = {
        task: sum(sample["task"] == task for sample in samples)
        for task in sorted(split["tasks"])
    }
    payload = {
        "schema_version": 1,
        "eval_seed": args.eval_seed,
        "batch_size": 16,
        "action_horizon": args.action_horizon,
        "samples_per_episode": args.samples_per_episode,
        "split_path": str(split_path),
        "split_sha256": sha256(split_path),
        "pairs_path": str(pairs_path),
        "pairs_sha256": sha256(pairs_path),
        "episodes_jsonl_path": str(episodes_path),
        "episodes_jsonl_sha256": sha256(episodes_path),
        "sample_count": len(samples),
        "sample_count_by_task": by_task,
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"samples": len(samples), "by_task": by_task}, sort_keys=True))


if __name__ == "__main__":
    main()

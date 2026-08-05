#!/usr/bin/env python3
"""Build a frozen episode-level split for milestone-stage tracker training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_task_episodes(
    episode_stages: dict[int, set[int]],
    *,
    val_count: int,
    seed: int,
    max_attempts: int = 100_000,
) -> tuple[list[int], list[int], int]:
    episodes = np.asarray(sorted(episode_stages), dtype=np.int64)
    if not 0 < val_count < episodes.size:
        raise ValueError(f"val_count={val_count} is invalid for {episodes.size} episodes")

    stage_episodes: dict[int, set[int]] = {}
    for episode, stages in episode_stages.items():
        for stage in stages:
            stage_episodes.setdefault(stage, set()).add(episode)
    singleton_episodes = {
        next(iter(stage_members))
        for stage_members in stage_episodes.values()
        if len(stage_members) == 1
    }
    eligible_for_val = np.asarray(
        [episode for episode in episodes if int(episode) not in singleton_episodes], dtype=np.int64
    )
    if eligible_for_val.size < val_count:
        raise ValueError(
            f"only {eligible_for_val.size} episodes are validation-eligible after preserving singleton stages"
        )

    rng = np.random.default_rng(seed)
    all_episodes = set(int(value) for value in episodes)
    for attempt in range(1, max_attempts + 1):
        val = set(int(value) for value in rng.choice(eligible_for_val, size=val_count, replace=False))
        train = all_episodes - val
        valid = True
        for members in stage_episodes.values():
            if not (members & train):
                valid = False
                break
            if len(members) >= 2 and not (members & val):
                valid = False
                break
        if valid:
            return sorted(train), sorted(val), attempt
    raise RuntimeError(f"failed to find a stage-covered split after {max_attempts} attempts")


def build_manifest(
    pairs_path: Path,
    task_map_path: Path,
    *,
    train_per_task: int,
    seed: int,
) -> dict[str, object]:
    pairs = np.load(pairs_path)
    required = {"cur_ep", "cur_ms", "pair_task"}
    missing = required.difference(pairs.files)
    if missing:
        raise ValueError(f"pairs archive is missing {sorted(missing)}")

    episode = np.asarray(pairs["cur_ep"], dtype=np.int64)
    stage = np.asarray(pairs["cur_ms"], dtype=np.int64)
    task = np.asarray(pairs["pair_task"], dtype=np.int64)
    task_map = json.loads(task_map_path.read_text())
    id_to_name = {int(task_id): name for name, task_id in task_map.items()}
    if set(np.unique(task)) != set(id_to_name):
        raise ValueError("task map does not match pair_task IDs")

    tasks: dict[str, object] = {}
    all_train: list[int] = []
    all_val: list[int] = []
    for task_id in sorted(id_to_name):
        mask = task == task_id
        episode_stages: dict[int, set[int]] = {}
        for ep, stage_id in zip(episode[mask], stage[mask], strict=True):
            episode_stages.setdefault(int(ep), set()).add(int(stage_id))
        val_count = len(episode_stages) - train_per_task
        train, val, attempts = split_task_episodes(
            episode_stages,
            val_count=val_count,
            seed=seed + task_id * 10_007,
        )
        stage_support = {}
        for stage_id in sorted(set(int(value) for value in stage[mask])):
            members = {ep for ep, stages in episode_stages.items() if stage_id in stages}
            stage_support[str(stage_id)] = {
                "episodes_total": len(members),
                "episodes_train": len(members.intersection(train)),
                "episodes_val": len(members.intersection(val)),
                "frame_rows": int(np.sum(mask & (stage == stage_id))),
                "validation_evaluable": len(members) >= 2,
            }
        tasks[id_to_name[task_id]] = {
            "task_id": task_id,
            "train_episodes": train,
            "val_episodes": val,
            "search_attempts": attempts,
            "stage_support": stage_support,
        }
        all_train.extend(train)
        all_val.extend(val)

    if set(all_train).intersection(all_val):
        raise AssertionError("episode leakage between train and validation")
    return {
        "version": "robotwin-mt-stage-tracker-split-v1",
        "seed": seed,
        "split_unit": "episode",
        "selection_rule": (
            "task-stratified random split; singleton-stage episodes remain in train; "
            "every stage supported by at least two episodes appears in both train and validation"
        ),
        "train_per_task": train_per_task,
        "val_per_task": len(all_val) // len(tasks),
        "train_episodes": sorted(all_train),
        "val_episodes": sorted(all_val),
        "tasks": tasks,
        "source": {
            "pairs_path": str(pairs_path),
            "pairs_sha256": sha256(pairs_path),
            "task_map_path": str(task_map_path),
            "task_map_sha256": sha256(task_map_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--task-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    manifest = build_manifest(
        args.pairs,
        args.task_map,
        train_per_task=args.train_per_task,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.output}: train={len(manifest['train_episodes'])} "
        f"val={len(manifest['val_episodes'])}"
    )


if __name__ == "__main__":
    main()

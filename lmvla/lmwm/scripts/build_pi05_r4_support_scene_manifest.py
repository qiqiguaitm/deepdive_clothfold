#!/usr/bin/env python3
"""Build a deterministic train-only R4 outcome-support supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TRAIN_EVAL_SEEDS = (0, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: dict, base: dict, *, task: str) -> dict:
    selected: dict[str, dict[str, list[int]]] = {}
    counts: set[int] = set()
    for eval_seed in TRAIN_EVAL_SEEDS:
        seed = str(eval_seed)
        source_scenes = [int(value) for value in source["eval_seeds"][seed][task]]
        base_scenes = {int(value) for value in base["eval_seeds"][seed][task]}
        supplement = [value for value in source_scenes if value not in base_scenes]
        if not supplement:
            raise ValueError(f"no unused scenes for eval_seed={eval_seed} task={task}")
        if len(supplement) + len(base_scenes) != len(source_scenes):
            raise ValueError(f"base scenes are not a subset for eval_seed={eval_seed} task={task}")
        if len(supplement) != len(set(supplement)):
            raise ValueError(f"duplicate supplemental scenes for eval_seed={eval_seed} task={task}")
        selected[seed] = {task: supplement}
        counts.add(len(supplement))
    if len(counts) != 1:
        raise ValueError(f"supplemental cell sizes differ: {sorted(counts)}")
    episodes_per_cell = counts.pop()
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_outcome_scene_seeds_v1",
        "amendment_protocol": "pi05_r4_train_outcome_support_supplement_v1",
        "selection_rule": "all source scenes not present in the frozen base manifest",
        "episodes_per_cell": episodes_per_cell,
        "split_by_eval_seed": {"train": list(TRAIN_EVAL_SEEDS), "eval": []},
        "task": task,
        "eval_seeds": selected,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        json.loads(args.source.read_text()),
        json.loads(args.base.read_text()),
        task=args.task,
    )
    payload["source_manifest_sha256"] = sha256(args.source)
    payload["base_manifest_sha256"] = sha256(args.base)
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

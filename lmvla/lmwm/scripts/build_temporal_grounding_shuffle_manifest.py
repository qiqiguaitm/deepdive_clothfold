#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    eval_seeds = payload["eval_seeds"]
    tasks = [str(task) for task in payload["tasks"]]
    expected = int(payload["episodes_per_cell"])
    assert expected >= 2

    mapping: dict[str, dict[str, dict[str, int]]] = {task: {} for task in tasks}
    for eval_seed in sorted(eval_seeds, key=int):
        task_map = eval_seeds[eval_seed]
        assert set(task_map) == set(tasks)
        for task in tasks:
            seeds = [int(seed) for seed in task_map[task]]
            assert len(seeds) == expected
            assert len(set(seeds)) == expected
            # A fixed one-position cyclic permutation is independent of all
            # rollout outcomes and guarantees a different episode in-task.
            sources = seeds[1:] + seeds[:1]
            pairs = {str(target): source_seed for target, source_seed in zip(seeds, sources)}
            assert all(int(target) != source_seed for target, source_seed in pairs.items())
            assert set(map(int, pairs)) == set(pairs.values()) == set(seeds)
            mapping[task][str(eval_seed)] = pairs

    result = {
        "schema_version": 1,
        "protocol": "temporal_grounding_tg1a_within_task_cyclic_v1",
        "frozen": True,
        "source_scene_manifest": str(args.source),
        "source_scene_manifest_sha256": sha256(source),
        "algorithm": "preserve source list order and map each scene to the next scene, wrapping once",
        "query_alignment": "source_query_index = target_query_index modulo frozen source query count",
        "outcome_independent": True,
        "self_matches": 0,
        "tasks": tasks,
        "eval_seeds": sorted(map(int, eval_seeds)),
        "episodes_per_cell": expected,
        "mapping": mapping,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

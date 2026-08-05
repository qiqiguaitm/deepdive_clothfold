#!/usr/bin/env python3
"""Build the audited R4 action-bearing outcome manifest from collector output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def build(
    result_root: Path,
    scene_manifest: dict,
    behavior_policy: Path,
    output_path: Path,
) -> dict:
    split_by_seed = {
        int(seed): split
        for split, seeds in scene_manifest["split_by_eval_seed"].items()
        for seed in seeds
    }
    expected = {
        (int(eval_seed), str(task)): [int(value) for value in values]
        for eval_seed, tasks in scene_manifest["eval_seeds"].items()
        for task, values in tasks.items()
    }
    summaries = sorted(result_root.glob("seed*/**/tasks/*/summary.json"))
    if len(summaries) != len(expected):
        raise ValueError(f"expected {len(expected)} summaries, found {len(summaries)}")

    records = []
    seen_cells = set()
    policy_digest = sha256(behavior_policy)
    manifest_root = output_path.parent
    for summary_path in summaries:
        seed_part = next(part for part in summary_path.parts if part.startswith("seed"))
        eval_seed = int(seed_part.removeprefix("seed"))
        payload = json.loads(summary_path.read_text())
        task = str(payload["task_name"])
        cell = (eval_seed, task)
        if cell not in expected or cell in seen_cells:
            raise ValueError(f"unexpected or duplicate cell: {cell}")
        seen_cells.add(cell)
        episodes = payload.get("episodes", [])
        actual_scenes = [int(episode["seed"]) for episode in episodes]
        if actual_scenes != expected[cell]:
            raise ValueError(f"scene identity mismatch for {cell}")
        for episode in episodes:
            local_episode_id = int(episode["episode_id"])
            trajectory = Path(str(episode.get("trajectory", "")))
            if not trajectory.is_absolute():
                trajectory = summary_path.parent / trajectory
            video = summary_path.parent / f"episode{local_episode_id}.mp4"
            if not trajectory.is_file() or not video.is_file():
                raise FileNotFoundError(f"missing trajectory/video for {cell} episode {local_episode_id}")
            records.append(
                {
                    "task": task,
                    "split": split_by_seed[eval_seed],
                    "eval_seed": eval_seed,
                    "episode_id": eval_seed * 100000 + local_episode_id,
                    "scene_seed": int(episode["seed"]),
                    "success": bool(episode["success"]),
                    "behavior_policy_sha256": policy_digest,
                    "trajectory": relative(trajectory, manifest_root),
                    "trajectory_sha256": sha256(trajectory),
                    "video": relative(video, manifest_root),
                    "video_sha256": sha256(video),
                }
            )
    if seen_cells != set(expected):
        raise ValueError("collector output is missing expected cells")
    return {
        "schema_version": 1,
        "protocol": "pi05_r4_action_bearing_outcomes_v1",
        "scene_manifest_protocol": scene_manifest["protocol"],
        "behavior_policy": str(behavior_policy.resolve()),
        "behavior_policy_sha256": policy_digest,
        "records": records,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--behavior-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.result_root.resolve(),
        json.loads(args.scene_manifest.read_text()),
        args.behavior_policy.resolve(),
        args.output.resolve(),
    )
    atomic_json(args.output, result)
    print(json.dumps({"records": len(result["records"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

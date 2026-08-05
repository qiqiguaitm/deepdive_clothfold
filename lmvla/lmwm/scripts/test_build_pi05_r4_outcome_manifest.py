from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_pi05_r4_outcome_manifest import build


def test_build_outcome_manifest_hashes_aligned_artifacts(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    scene_manifest = {
        "protocol": "scenes",
        "split_by_eval_seed": {"train": [0], "eval": [1]},
        "eval_seeds": {
            "0": {"task": [10]},
            "1": {"task": [20]},
        },
    }
    for eval_seed, scene_seed in ((0, 10), (1, 20)):
        task_dir = result_root / f"seed{eval_seed}" / "run" / "tasks" / "task"
        task_dir.mkdir(parents=True)
        trajectory = task_dir / "episode0.npz"
        np.savez_compressed(
            trajectory,
            actions=np.zeros((2, 14)),
            states=np.zeros((2, 14)),
            frame_index=np.asarray([0, 1]),
        )
        (task_dir / "episode0.mp4").write_bytes(b"video")
        (task_dir / "summary.json").write_text(
            json.dumps(
                {
                    "task_name": "task",
                    "episodes": [
                        {
                            "episode_id": 0,
                            "seed": scene_seed,
                            "success": eval_seed == 0,
                            "trajectory": str(trajectory),
                        }
                    ],
                }
            )
        )
    policy = tmp_path / "model.safetensors"
    policy.write_bytes(b"policy")
    output = tmp_path / "manifest.json"

    result = build(result_root, scene_manifest, policy, output)

    assert len(result["records"]) == 2
    assert {record["split"] for record in result["records"]} == {"train", "eval"}
    assert len({record["behavior_policy_sha256"] for record in result["records"]}) == 1
    assert all(len(record["trajectory_sha256"]) == 64 for record in result["records"])


def test_build_outcome_manifest_accepts_explicit_predeclared_shard(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    scene_manifest = {
        "protocol": "scenes",
        "split_by_eval_seed": {"train": [0, 1], "eval": [2, 3]},
        "eval_seeds": {
            str(seed): {
                "task_a": [seed * 10 + 1],
                "task_b": [seed * 10 + 2],
            }
            for seed in range(4)
        },
    }
    for eval_seed in (0, 1):
        task_dir = result_root / f"seed{eval_seed}" / "run" / "tasks" / "task_b"
        task_dir.mkdir(parents=True)
        trajectory = task_dir / "episode0.npz"
        np.savez_compressed(
            trajectory,
            actions=np.zeros((2, 14)),
            states=np.zeros((2, 14)),
            frame_index=np.asarray([0, 1]),
        )
        (task_dir / "episode0.mp4").write_bytes(b"video")
        (task_dir / "summary.json").write_text(
            json.dumps(
                {
                    "task_name": "task_b",
                    "episodes": [
                        {
                            "episode_id": 0,
                            "seed": eval_seed * 10 + 2,
                            "success": bool(eval_seed),
                            "trajectory": str(trajectory),
                        }
                    ],
                }
            )
        )
    policy = tmp_path / "model.safetensors"
    policy.write_bytes(b"policy")
    output = tmp_path / "manifest.json"

    result = build(
        result_root,
        scene_manifest,
        policy,
        output,
        tasks={"task_b"},
        eval_seeds={0, 1},
    )

    assert result["selected_tasks"] == ["task_b"]
    assert result["selected_eval_seeds"] == [0, 1]
    assert len(result["records"]) == 2

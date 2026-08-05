from __future__ import annotations

import hashlib
import json

import numpy as np

from audit_pi05_r4_outcome_dataset import REQUIRED_TASKS, audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(tmp_path, *, leak=False, omit_actions=False):
    records = []
    for task_index, task in enumerate(sorted(REQUIRED_TASKS)):
        for split_index, split in enumerate(("train", "eval")):
            outcomes = (False, True) if split == "train" else (True,)
            for outcome_index, success in enumerate(outcomes):
                identity = f"{task_index}_{split}_{outcome_index}"
                trajectory = tmp_path / f"{identity}.npz"
                arrays = {
                    "states": np.ones((3, 14), dtype=np.float32),
                    "frame_index": np.arange(3, dtype=np.int32),
                }
                if not omit_actions:
                    arrays["actions"] = np.ones((3, 14), dtype=np.float32)
                np.savez(trajectory, **arrays)
                video = tmp_path / f"{identity}.mp4"
                video.write_bytes(b"video")
                scene_seed = task_index * 100 + split_index * 10 + outcome_index
                if leak and split == "eval":
                    scene_seed = task_index * 100
                records.append(
                    {
                        "task": task,
                        "split": split,
                        "episode_id": split_index * 10 + outcome_index,
                        "scene_seed": scene_seed,
                        "success": success,
                        "behavior_policy_sha256": "a" * 64,
                        "trajectory": trajectory.name,
                        "trajectory_sha256": digest(trajectory),
                        "video": video.name,
                        "video_sha256": digest(video),
                    }
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records": records}))
    return manifest


def test_accepts_action_bearing_disjoint_six_task_dataset(tmp_path):
    result = audit(build_manifest(tmp_path))
    assert result["accepted"]
    assert result["transition_count"] > 0


def test_rejects_scene_leakage(tmp_path):
    result = audit(build_manifest(tmp_path, leak=True))
    assert not result["accepted"]
    assert not result["checks"]["train_eval_scene_disjoint"]


def test_rejects_video_only_rollouts_without_actions(tmp_path):
    result = audit(build_manifest(tmp_path, omit_actions=True))
    assert not result["accepted"]
    assert not result["checks"]["action_state_observation_alignment_present"]

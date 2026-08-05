from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_pi05_r4_query_dataset import audit
from build_pi05_r4_query_manifest import build
from merge_pi05_r4_query_manifests import merge


TASKS = (
    "beat_block_hammer",
    "blocks_ranking_size",
    "blocks_ranking_rgb",
    "handover_block",
    "stack_blocks_two",
    "stack_blocks_three",
)
POLICY = "a" * 64


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_episode(root: Path, episode: int, value: int) -> tuple[Path, Path, Path]:
    trajectory = root / f"episode{episode}.npz"
    queries = root / f"query_episode{episode}.npz"
    video = root / f"episode{episode}.mp4"
    states = np.tile(np.arange(14, dtype=np.float32), (75, 1))
    np.savez_compressed(
        trajectory,
        actions=states + 0.25,
        states=states,
        frame_index=np.arange(75, dtype=np.int64),
    )
    image = np.full((2, 4, 6, 3), value, dtype=np.uint8)
    np.savez_compressed(
        queries,
        query_frame_index=np.asarray([0, 50], dtype=np.int64),
        query_states=states[[0, 50]],
        cam_high=image,
        cam_left_wrist=image + 1,
        cam_right_wrist=image + 2,
        instruction=np.asarray("perform the task"),
    )
    video.write_bytes(b"video")
    return trajectory, queries, video


def outcome_record(root: Path, task: str, episode: int, success: bool) -> dict:
    trajectory, _, video = write_episode(root, episode, episode)
    return {
        "task": task,
        "split": "train",
        "eval_seed": episode % 2,
        "episode_id": episode,
        "scene_seed": 1000 + episode,
        "success": success,
        "behavior_policy_sha256": POLICY,
        "trajectory": trajectory.name,
        "trajectory_sha256": digest(trajectory),
        "video": video.name,
        "video_sha256": digest(video),
    }


def test_build_merge_and_audit_query_manifests(tmp_path):
    source_manifests = []
    all_outcomes = []
    for source_index in range(2):
        source_root = tmp_path / f"source{source_index}"
        source_root.mkdir()
        records = []
        for task_index, task in enumerate(TASKS):
            episode = source_index * len(TASKS) + task_index
            records.append(outcome_record(source_root, task, episode, bool(source_index)))
        outcome = {
            "schema_version": 1,
            "protocol": "pi05_r4_action_bearing_outcomes_v1",
            "behavior_policy_sha256": POLICY,
            "records": records,
        }
        outcome_path = source_root / "dataset_manifest.json"
        outcome_path.write_text(json.dumps(outcome))
        query_path = source_root / "query_manifest.json"
        query_path.write_text(json.dumps(build(outcome_path, query_path)))
        source_manifests.append(query_path)
        all_outcomes.extend(records)

    combined_query_path = tmp_path / "combined_query.json"
    combined_query_path.write_text(
        json.dumps(merge(source_manifests, combined_query_path))
    )
    combined_outcome_path = tmp_path / "combined_outcome.json"
    combined_outcome_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "pi05_r4_action_bearing_outcomes_combined_v1",
                "behavior_policy_sha256": POLICY,
                "records": [
                    {
                        **record,
                        "trajectory": str(
                            (tmp_path / f"source{index // len(TASKS)}" / record["trajectory"]).resolve()
                        ),
                        "video": str(
                            (tmp_path / f"source{index // len(TASKS)}" / record["video"]).resolve()
                        ),
                    }
                    for index, record in enumerate(all_outcomes)
                ],
            }
        )
    )

    report = audit(combined_query_path, combined_outcome_path)
    assert report["accepted"]
    assert report["record_count"] == 12
    assert report["query_count"] == 24
    assert all(value == {"success": 1, "failure": 1} for value in report["support"].values())


def test_audit_rejects_missing_predeclared_scene(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    record = outcome_record(root, TASKS[0], 0, False)
    outcome = {"behavior_policy_sha256": POLICY, "records": [record]}
    outcome_path = root / "outcome.json"
    outcome_path.write_text(json.dumps(outcome))
    query_path = root / "query.json"
    query_path.write_text(json.dumps(build(outcome_path, query_path)))
    payload = json.loads(query_path.read_text())
    payload["records"] = []
    query_path.write_text(json.dumps(payload))

    report = audit(query_path, outcome_path)
    assert not report["accepted"]
    assert not report["checks"]["exact_predeclared_train_scene_outcomes"]

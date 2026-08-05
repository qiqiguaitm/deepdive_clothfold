from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("merge_pi05_r4_outcome_manifests.py")
SPEC = importlib.util.spec_from_file_location("merge_pi05_r4_outcome_manifests", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_manifest(path: Path, *, policy: str, scene: int, episode: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "behavior_policy_sha256": policy,
                "records": [
                    {
                        "task": "task",
                        "split": "train",
                        "episode_id": episode,
                        "scene_seed": scene,
                        "trajectory": "trajectory.npz",
                        "video": "video.mp4",
                    }
                ],
            }
        )
    )


def test_merge_reindexes_and_rebases_artifact_paths(tmp_path: Path) -> None:
    policy = "a" * 64
    first = tmp_path / "first" / "manifest.json"
    second = tmp_path / "second" / "manifest.json"
    output = tmp_path / "combined" / "manifest.json"
    write_manifest(first, policy=policy, scene=10)
    write_manifest(second, policy=policy, scene=11)

    result = MODULE.merge([first, second], output)

    assert [record["episode_id"] for record in result["records"]] == [0, 1]
    assert result["records"][0]["trajectory"] == "../first/trajectory.npz"
    assert result["records"][1]["video"] == "../second/video.mp4"
    assert result["behavior_policy_sha256"] == policy


def test_merge_rejects_duplicate_scene(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_manifest(first, policy="a" * 64, scene=10)
    write_manifest(second, policy="a" * 64, scene=10)
    with pytest.raises(ValueError, match="duplicate task/scene"):
        MODULE.merge([first, second], tmp_path / "out.json")


def test_merge_rejects_policy_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_manifest(first, policy="a" * 64, scene=10)
    write_manifest(second, policy="b" * 64, scene=11)
    with pytest.raises(ValueError, match="behavior policy mismatch"):
        MODULE.merge([first, second], tmp_path / "out.json")

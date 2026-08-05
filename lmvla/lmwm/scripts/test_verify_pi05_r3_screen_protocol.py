import json
from pathlib import Path

import numpy as np
import pytest

from verify_pi05_r3_screen_protocol import sha256, verify


def test_verifier_rejects_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    scene = tmp_path / "lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "conditions": {"semantic_next": {}},
        "source_sha256": {"source.py": sha256(source)},
        "scene_manifest_sha256": sha256(scene),
    }))
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="source drift"):
        verify(tmp_path, protocol, tmp_path / "artifact", "semantic_next")


def test_verifier_accepts_complete_minimal_artifact(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    scene = tmp_path / "lmvla/lmwm/data/pi05_r3_semantic_screen_scene_seeds_v1.json"
    scene.parent.mkdir(parents=True)
    scene.write_text("{}\n")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    vocabulary = [{"task": "task", "local_id": index, "name": f"event_{index}"} for index in range(16)]
    (artifact / "vocabulary.json").write_text(json.dumps(vocabulary))
    (artifact / "segments.jsonl").write_text("{}\n")
    (artifact / "semantic_profile_episodes.jsonl").write_text('{"episode_index":0,"length":2}\n')
    (artifact / "task_map.json").write_text('{"task":0}\n')
    np.savez_compressed(artifact / "semantic_profile_pairs.npz", pair_task=[0,0], cur_ep=[0,0], cur_fi=[0,1], cur_ms=[0,1])
    files = {
        "vocabulary": artifact / "vocabulary.json",
        "segments": artifact / "segments.jsonl",
        "semantic_profile_pairs": artifact / "semantic_profile_pairs.npz",
        "semantic_profile_episodes": artifact / "semantic_profile_episodes.jsonl",
        "task_map": artifact / "task_map.json",
    }
    manifest = {
        "semantic_event_count": 16,
        "semantic_profile_frame_count": 2,
        "semantic_profile_episode_count": 1,
        "tasks": {"task": {}},
        **{f"{name}_sha256": sha256(path) for name, path in files.items()},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "conditions": {"semantic_next": {"prompt_mode": "semantic-next"}},
        "source_sha256": {"source.py": sha256(source)},
        "scene_manifest_sha256": sha256(scene),
    }))
    result = verify(tmp_path, protocol, artifact, "semantic_next")
    assert result["accepted"] is True
    assert result["semantic_profile_frames"] == 2

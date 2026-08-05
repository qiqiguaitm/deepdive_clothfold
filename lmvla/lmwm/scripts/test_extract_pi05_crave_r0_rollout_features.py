import json
from pathlib import Path

import cv2
import numpy as np

from extract_pi05_crave_r0_rollout_features import decode_video, discover_episodes


def test_discovery_and_strided_decode(tmp_path: Path) -> None:
    task = tmp_path / "seed1" / "run" / "tag" / "tasks" / "stack_blocks_two"
    task.mkdir(parents=True)
    (task / "summary.json").write_text(
        json.dumps(
            {
                "task_name": "stack_blocks_two",
                "episodes": [
                    {"episode_id": 3, "seed": 1003, "success": False, "steps": 7}
                ],
            }
        ),
        encoding="utf-8",
    )
    video = task / "episode3.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 12)
    )
    for index in range(7):
        writer.write(np.full((12, 16, 3), index * 20, dtype=np.uint8))
    writer.release()

    records = discover_episodes(tmp_path)
    assert len(records) == 1
    assert records[0]["simulator_seed"] == 1
    assert records[0]["scene_seed"] == 1003
    frames, indices, decoded = decode_video(video, stride=3)
    assert decoded == 7
    assert len(frames) == 3
    np.testing.assert_array_equal(indices, [0, 3, 6])

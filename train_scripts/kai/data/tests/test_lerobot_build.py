from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_build import CanonicalBuildSpec  # noqa: E402
from lerobot_build import build_canonical_split  # noqa: E402
from lerobot_build import read_jsonl  # noqa: E402


CAMERAS = ("observation.images.top", "observation.images.left")


def _write_source(root: Path) -> list[dict]:
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    rows = []
    for episode, length in enumerate((2, 3)):
        vector = [np.arange(14, dtype=np.float32) + frame for frame in range(length)]
        frame = pd.DataFrame(
            {
                "observation.state": vector,
                "action": vector,
                "frame_index": np.arange(length),
                "episode_index": episode,
                "index": np.arange(length),
                "task_index": 7,
            }
        )
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet",
        )
        for camera in CAMERAS:
            path = root / "videos" / "chunk-000" / camera / f"episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"video-{episode}-{camera}".encode())
        rows.append({"episode_index": episode, "length": length, "tasks": ["old"]})

    info = {"features": {"observation.state": {}, "action": {}, "depth": {}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text('{"task_index":7,"task":"old"}\n')
    return rows


def test_build_canonical_split_reindexes_and_creates_relocatable_links(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    rows = _write_source(source)

    result = build_canonical_split(
        CanonicalBuildSpec(
            source_root=source,
            output_root=output,
            cameras=CAMERAS,
            action_width=14,
            task="fold cloth",
            drop_features=("depth",),
        ),
        list(reversed(rows)),
        metadata_transform=lambda row, old, _new: {**row, "source_episode_index": old},
    )

    assert (result.episodes, result.frames, result.videos) == (2, 5, 4)
    first = pq.read_table(output / "data" / "chunk-000" / "episode_000000.parquet").to_pandas()
    second = pq.read_table(output / "data" / "chunk-000" / "episode_000001.parquet").to_pandas()
    assert first["episode_index"].unique().tolist() == [0]
    assert second["episode_index"].unique().tolist() == [1]
    assert first["index"].tolist() == [0, 1, 2]
    assert second["index"].tolist() == [3, 4]
    assert first["task_index"].unique().tolist() == [0]

    link = output / "videos" / "chunk-000" / CAMERAS[0] / "episode_000000.mp4"
    assert link.is_symlink()
    assert link.resolve() == source / "videos" / "chunk-000" / CAMERAS[0] / "episode_000001.mp4"

    info = json.loads((output / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["total_videos"] == 4
    assert "depth" not in info["features"]
    assert read_jsonl(output / "meta" / "episodes.jsonl")[0]["source_episode_index"] == 1
    assert read_jsonl(output / "meta" / "tasks.jsonl") == [{"task_index": 0, "task": "fold cloth"}]
    stats = read_jsonl(output / "meta" / "episodes_stats.jsonl")
    assert len(stats) == 2
    assert stats[0]["stats"]["action"]["count"] == [3]


def test_build_canonical_split_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    spec = CanonicalBuildSpec(tmp_path / "source", output, CAMERAS, 14)

    try:
        build_canonical_split(spec, [])
    except FileExistsError as error:
        assert str(output) in str(error)
    else:
        raise AssertionError("existing output must not be overwritten")

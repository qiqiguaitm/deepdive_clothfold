"""Reusable primitives for building canonical single-chunk LeRobot datasets.

Dataset-specific scripts should decide *which* episodes belong in each split.
This module owns the mechanical and easy-to-get-wrong part: validation,
reindexing, metadata/stat generation, and relocatable video materialization.

It intentionally does not discover raw collection layouts or encode policy
decisions such as date filters, quality thresholds, or train/val allocation.
Those choices belong in a small wrapper or a local build specification.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VideoMode = Literal["relative_symlink", "copy"]
MetadataTransform = Callable[[Mapping[str, Any], int, int], dict[str, Any]]


@dataclass(frozen=True)
class CanonicalBuildSpec:
    source_root: Path
    output_root: Path
    cameras: tuple[str, ...]
    action_width: int
    video_mode: VideoMode = "relative_symlink"
    source_chunk: str = "chunk-000"
    output_chunk: str = "chunk-000"
    state_column: str = "observation.state"
    action_column: str = "action"
    task: str | None = None
    drop_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildResult:
    episodes: int
    frames: int
    videos: int
    metadata: tuple[dict[str, Any], ...]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")


def per_episode_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Create LeRobot scalar/array episode statistics; image stats are omitted."""
    stats: dict[str, Any] = {}
    for column in frame.columns:
        values = frame[column].to_numpy()
        if values.dtype == object:
            array = np.stack(values).astype(np.float64)
        else:
            array = values.astype(np.float64).reshape(len(values), -1)
        stats[column] = {
            "mean": array.mean(0).tolist(),
            "std": array.std(0).tolist(),
            "min": array.min(0).tolist(),
            "max": array.max(0).tolist(),
            "count": [len(array)],
        }
    return stats


def _validate_vector_column(frame: pd.DataFrame, column: str, width: int, episode: int) -> None:
    if column not in frame:
        raise ValueError(f"episode {episode}: missing column {column!r}")
    values = np.stack(frame[column].to_numpy())
    if values.shape != (len(frame), width):
        raise ValueError(f"episode {episode}: {column} has shape {values.shape}, expected {(len(frame), width)}")
    if not np.isfinite(values).all():
        raise ValueError(f"episode {episode}: {column} contains NaN/Inf")


def _materialize_video(source: Path, target: Path, mode: VideoMode) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "relative_symlink":
        target.symlink_to(os.path.relpath(source, target.parent))
    elif mode == "copy":
        shutil.copy2(source.resolve(), target)
    else:  # pragma: no cover - protected by the VideoMode type for typed callers
        raise ValueError(f"unsupported video mode: {mode}")


def build_canonical_split(
    spec: CanonicalBuildSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_index_key: str = "episode_index",
    expected_lengths: bool = True,
    metadata_transform: MetadataTransform | None = None,
) -> BuildResult:
    """Build one reindexed split from a canonical single-chunk source.

    ``rows`` controls selection and ordering. Every row must contain
    ``source_index_key`` and, when ``expected_lengths`` is true, ``length``.
    The output directory must not exist; callers should build in a staging
    directory and rename it only after all split-level checks pass.
    """
    if spec.output_root.exists():
        raise FileExistsError(f"output already exists: {spec.output_root}")
    (spec.output_root / "data" / spec.output_chunk).mkdir(parents=True)
    (spec.output_root / "meta").mkdir(parents=True)

    metadata: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    global_index = 0

    for new_episode, source_row in enumerate(rows):
        source_episode = int(source_row[source_index_key])
        source_parquet = (
            spec.source_root
            / "data"
            / spec.source_chunk
            / f"episode_{source_episode:06d}.parquet"
        )
        frame = pq.read_table(source_parquet).to_pandas()
        length = len(frame)
        if expected_lengths and length != int(source_row["length"]):
            raise ValueError(
                f"episode {source_episode}: metadata length {source_row['length']} != parquet length {length}"
            )
        _validate_vector_column(frame, spec.state_column, spec.action_width, source_episode)
        _validate_vector_column(frame, spec.action_column, spec.action_width, source_episode)

        frame["frame_index"] = np.arange(length, dtype=np.int64)
        frame["episode_index"] = np.int64(new_episode)
        frame["index"] = np.arange(global_index, global_index + length, dtype=np.int64)
        frame["task_index"] = np.int64(0)
        output_parquet = (
            spec.output_root
            / "data"
            / spec.output_chunk
            / f"episode_{new_episode:06d}.parquet"
        )
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), output_parquet)

        for camera in spec.cameras:
            source_video = (
                spec.source_root
                / "videos"
                / spec.source_chunk
                / camera
                / f"episode_{source_episode:06d}.mp4"
            )
            target_video = (
                spec.output_root
                / "videos"
                / spec.output_chunk
                / camera
                / f"episode_{new_episode:06d}.mp4"
            )
            _materialize_video(source_video, target_video, spec.video_mode)

        if metadata_transform is None:
            item = dict(source_row)
        else:
            item = metadata_transform(source_row, source_episode, new_episode)
        item["episode_index"] = new_episode
        item["length"] = length
        if spec.task is not None:
            item["tasks"] = [spec.task]
        metadata.append(item)
        episode_stats.append({"episode_index": new_episode, "stats": per_episode_stats(frame)})
        global_index += length

    info = json.loads((spec.source_root / "meta" / "info.json").read_text())
    for feature in spec.drop_features:
        info.get("features", {}).pop(feature, None)
    info.update(
        total_episodes=len(rows),
        total_frames=global_index,
        total_tasks=1,
        total_videos=len(rows) * len(spec.cameras),
        total_chunks=1,
        chunks_size=max(1000, len(rows)),
        splits={"train": f"0:{len(rows)}"},
    )
    (spec.output_root / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_jsonl(spec.output_root / "meta" / "episodes.jsonl", metadata)
    write_jsonl(spec.output_root / "meta" / "episodes_stats.jsonl", episode_stats)
    if spec.task is None:
        shutil.copy2(spec.source_root / "meta" / "tasks.jsonl", spec.output_root / "meta" / "tasks.jsonl")
    else:
        write_jsonl(spec.output_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": spec.task}])

    return BuildResult(
        episodes=len(rows),
        frames=global_index,
        videos=len(rows) * len(spec.cameras),
        metadata=tuple(metadata),
    )

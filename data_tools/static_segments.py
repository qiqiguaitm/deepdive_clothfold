"""Read-only detection of long stationary runs in LeRobot episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .lerobot import discover_episodes, write_jsonl

ARM_DIMS_14 = tuple(range(0, 6)) + tuple(range(7, 13))
GRIP_DIMS_14 = (6, 13)


@dataclass(frozen=True)
class StaticSegment:
    episode_id: int
    start_frame: int
    end_frame: int  # inclusive
    frames: int
    duration_s: float
    position: str  # leading | interior | trailing | whole
    ideal_only: bool


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous true ranges."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def detect_static_segments(
    values: np.ndarray,
    *,
    episode_id: int,
    fps: float = 30.0,
    min_frames: int = 50,
    arm_threshold: float = 3e-3,
    gripper_threshold: float = 0.02,
    ideal_mask: np.ndarray | None = None,
) -> list[StaticSegment]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"trajectory must be [T,D], got {values.shape}")
    if min_frames < 1:
        raise ValueError("min_frames must be >= 1")
    delta = np.zeros_like(values, dtype=np.float64)
    if len(values) > 1:
        delta[1:] = np.abs(np.diff(values.astype(np.float64), axis=0))
    if values.shape[1] >= 14:
        arm_static = np.all(delta[:, ARM_DIMS_14] < arm_threshold, axis=1)
        grip_static = np.all(delta[:, GRIP_DIMS_14] < gripper_threshold, axis=1)
        static = arm_static & grip_static
    else:
        static = np.all(delta < arm_threshold, axis=1)
    # Frame zero has no preceding transition and must not manufacture a static
    # prefix by itself; it joins a run only when frame one is also static.
    if len(static) == 1:
        static[0] = False
    else:
        static[0] = static[1]
    if ideal_mask is not None:
        if ideal_mask.shape != static.shape:
            raise ValueError("ideal mask length differs from trajectory")
        static &= ideal_mask

    total = len(static)
    result: list[StaticSegment] = []
    for start, end in _runs(static):
        length = end - start + 1
        if length < min_frames:
            continue
        if start == 0 and end == total - 1:
            position = "whole"
        elif start == 0:
            position = "leading"
        elif end == total - 1:
            position = "trailing"
        else:
            position = "interior"
        result.append(StaticSegment(
            episode_id=episode_id,
            start_frame=start,
            end_frame=end,
            frames=length,
            duration_s=round(length / fps, 6),
            position=position,
            ideal_only=ideal_mask is not None,
        ))
    return result


def scan_static_segments(
    root: Path,
    *,
    episodes: Iterable[int] | None = None,
    min_frames: int = 50,
    fps: float = 30.0,
    source_column: str = "observation.state",
    arm_threshold: float = 3e-3,
    gripper_threshold: float = 0.02,
    ideal_only: bool = False,
    ideal_classes: Iterable[int] = (1, 5),
    output: Path | None = None,
) -> list[StaticSegment]:
    import pyarrow.parquet as pq

    selected = discover_episodes(root) if episodes is None else sorted(set(episodes))
    segments: list[StaticSegment] = []
    ideal_values = set(ideal_classes)
    for episode in selected:
        matches = list(root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
        if len(matches) != 1:
            raise ValueError(f"episode {episode}: expected one parquet, got {len(matches)}")
        columns = [source_column]
        schema = pq.read_schema(matches[0])
        if ideal_only:
            if "dagger_frame_class" not in schema.names:
                raise ValueError(f"{matches[0]} has no dagger_frame_class for --ideal-only")
            columns.append("dagger_frame_class")
        table = pq.read_table(matches[0], columns=columns)
        trajectory = np.asarray(table[source_column].to_pylist(), dtype=np.float64)
        ideal_mask = None
        if ideal_only:
            classes = np.asarray(table["dagger_frame_class"].to_pylist(), dtype=np.int64)
            ideal_mask = np.isin(classes, list(ideal_values))
        segments.extend(detect_static_segments(
            trajectory,
            episode_id=episode,
            fps=fps,
            min_frames=min_frames,
            arm_threshold=arm_threshold,
            gripper_threshold=gripper_threshold,
            ideal_mask=ideal_mask,
        ))
    if output:
        write_jsonl(output, (asdict(segment) for segment in segments))
    return segments

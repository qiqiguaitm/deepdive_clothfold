"""Fast structural quality checks for KAI0/AgileX LeRobot episodes."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .lerobot import DEFAULT_CAMERAS, iter_episode_artifacts, write_jsonl


@dataclass
class EpisodeQuality:
    episode_id: int
    good: bool
    parquet_frames: int = 0
    video_frames: dict[str, int] | None = None
    reasons: list[str] | None = None


def _video_frames(path: Path) -> int:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,nb_frames", "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout).get("streams", [{}])[0]
    raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    return int(raw) if raw not in (None, "N/A") else 0


def inspect_episode(ep: int, parquet: Path, videos: dict[str, Path]) -> EpisodeQuality:
    import numpy as np
    import pyarrow.parquet as pq

    reasons: list[str] = []
    table = pq.read_table(parquet)
    frames = len(table)
    if frames <= 0:
        reasons.append("empty_parquet")
    if "action" not in table.column_names:
        reasons.append("missing_action")
    elif frames:
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if not np.isfinite(action).all():
            reasons.append("nonfinite_action")
        elif np.max(np.abs(action), initial=0.0) <= 1e-8:
            reasons.append("all_zero_action")

    counts: dict[str, int] = {}
    for camera, path in videos.items():
        if not path.is_file():
            reasons.append(f"missing_video:{camera}")
            counts[camera] = 0
            continue
        try:
            counts[camera] = _video_frames(path)
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
            counts[camera] = 0
            reasons.append(f"invalid_video:{camera}")
        if counts[camera] != frames:
            reasons.append(f"frame_mismatch:{camera}:{counts[camera]}!={frames}")
    return EpisodeQuality(ep, not reasons, frames, counts, reasons)


def scan_dataset(
    root: Path,
    *,
    episodes: Iterable[int] | None = None,
    cameras: Iterable[str] = DEFAULT_CAMERAS,
    output: Path | None = None,
) -> list[EpisodeQuality]:
    rows = [
        inspect_episode(ep, parquet, videos)
        for ep, parquet, videos in iter_episode_artifacts(root, episodes, cameras)
    ]
    if output:
        write_jsonl(output, (asdict(row) for row in rows))
        good = output.with_suffix(".good_episodes.txt")
        bad = output.with_suffix(".bad_episodes.txt")
        good.write_text("\n".join(f"{row.episode_id:06d}" for row in rows if row.good) + "\n")
        bad.write_text(
            "\n".join(
                f"{row.episode_id:06d}\t{';'.join(row.reasons or [])}"
                for row in rows if not row.good
            ) + "\n"
        )
    return rows

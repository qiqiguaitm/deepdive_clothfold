"""Shared LeRobot v2/v2.1 filesystem and metadata helpers.

This module consolidates the repeated primitives found in the historical KAI0
build scripts and the AgileX scripts archived on 2026-07-31.  Heavy dependencies
are imported lazily so inventory and Forge commands work in a plain Python env.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

EPISODE_RE = re.compile(r"episode_(\d+)\.(?:parquet|mp4)$")
DEFAULT_CAMERAS = ("top_head", "hand_left", "hand_right")
VideoMode = Literal["hardlink", "copy", "symlink"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def episode_id(path: Path) -> int:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"not an episode artifact: {path}")
    return int(match.group(1))


@dataclass(frozen=True)
class DatasetLayout:
    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta"

    def parquet(self, episode: int, chunk: int = 0) -> Path:
        return self.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"

    def video(self, camera: str, episode: int, chunk: int = 0) -> Path:
        return (
            self.root
            / "videos"
            / f"chunk-{chunk:03d}"
            / f"observation.images.{camera}"
            / f"episode_{episode:06d}.mp4"
        )


def discover_episodes(root: Path) -> list[int]:
    ids = {
        episode_id(path)
        for path in root.glob("data/chunk-*/episode_*.parquet")
        if EPISODE_RE.fullmatch(path.name)
    }
    return sorted(ids)


def parse_episode_file(path: Path) -> list[int]:
    """Read newline/CSV-like good-episode lists emitted by AgileX quality jobs."""
    result: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        token = raw.strip()
        if not token:
            continue
        try:
            result.add(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid episode id at {path}:{line_no}: {token!r}") from exc
    return sorted(result)


def latest_good_file(root: Path) -> Path | None:
    candidates = list((root / "meta" / "quality").glob("*.good_episodes.txt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def strip_depth_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if "depth" not in key.lower()}


def place_file(src: Path, dst: Path, mode: VideoMode = "hardlink") -> None:
    """Place an immutable artifact without silently overwriting an existing one."""
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown placement mode: {mode}")


def iter_episode_artifacts(
    root: Path,
    episodes: Iterable[int] | None = None,
    cameras: Iterable[str] = DEFAULT_CAMERAS,
) -> Iterator[tuple[int, Path, dict[str, Path]]]:
    layout = DatasetLayout(root)
    for ep in discover_episodes(root) if episodes is None else sorted(set(episodes)):
        parquet_candidates = list(root.glob(f"data/chunk-*/episode_{ep:06d}.parquet"))
        if len(parquet_candidates) != 1:
            raise ValueError(f"episode {ep}: expected one parquet, got {len(parquet_candidates)}")
        parquet = parquet_candidates[0]
        chunk = int(parquet.parent.name.split("-", 1)[1])
        videos: dict[str, Path] = {}
        for camera in cameras:
            canonical = layout.video(camera, ep, chunk)
            legacy = canonical.parent.parent / camera / canonical.name
            videos[camera] = canonical if canonical.exists() else legacy
        yield ep, parquet, videos


def rewrite_episode_table(table: Any, new_episode: int, global_offset: int, fps: int) -> Any:
    """Reindex a pyarrow episode table while preserving task-specific columns."""
    import numpy as np
    import pyarrow as pa

    length = len(table)
    values = {
        "episode_index": pa.array(np.full(length, new_episode, dtype=np.int64)),
        "frame_index": pa.array(np.arange(length, dtype=np.int64)),
        "index": pa.array(np.arange(global_offset, global_offset + length, dtype=np.int64)),
        "timestamp": pa.array(np.arange(length, dtype=np.float64) / float(fps)),
    }
    for name, array in values.items():
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, array) if index >= 0 else table.append_column(name, array)
    return table

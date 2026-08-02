#!/usr/bin/env python3
"""Build a LeRobot v2.1 RoboTwin mirror with official per-episode prompts.

The North RoboTwin v2.1 conversion keeps the correct observations and actions,
but assigns a randomly changing paraphrase task_index to every frame. The
released ``lerobot/robotwin_unified`` dataset assigns one task per episode.
This builder preserves the existing 50 Hz videos and timestamps, rewrites only
the task_index column in the small episode parquet files, and regenerates v2.1
metadata from the official v3 task and episode tables. The official release
encodes the same frame sequences at 30 Hz; changing metadata without re-encoding
the existing 50 Hz videos would make video timestamp queries incorrect.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-tasks", type=Path, required=True)
    parser.add_argument("--official-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Build only the first N episodes for a smoke test.",
    )
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("stats/"):
            continue
        _, feature, statistic = key.split("/", 2)
        result.setdefault(feature, {})[statistic] = value
    return result


def patched_stats(
    row: dict[str, Any],
    source_stats: dict[str, dict[str, Any]],
    task_index: int,
    length: int,
) -> dict[str, dict[str, Any]]:
    stats = stats_from_row(row)
    # Keep timestamp stats paired with the existing 50 Hz video encoding.
    stats["timestamp"] = source_stats["timestamp"]
    stats["task_index"] = {
        "min": [task_index],
        "max": [task_index],
        "mean": [float(task_index)],
        "std": [0.0],
        "count": [length],
    }
    return stats


def source_episode_path(source: Path, episode_index: int, chunk_size: int) -> Path:
    return (
        source
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def output_episode_path(output: Path, episode_index: int, chunk_size: int) -> Path:
    return (
        output
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def rewrite_episode(
    source: Path,
    output: Path,
    chunk_size: int,
    episode_index: int,
    task_index: int,
    expected_length: int,
) -> tuple[int, int]:
    src = source_episode_path(source, episode_index, chunk_size)
    dst = output_episode_path(output, episode_index, chunk_size)
    if dst.is_file():
        metadata = pq.read_metadata(dst)
        if metadata.num_rows == expected_length:
            task_values = pq.read_table(dst, columns=["task_index"])[
                "task_index"
            ].to_pylist()
            if set(task_values) == {task_index}:
                return episode_index, expected_length

    table = pq.read_table(src)
    if table.num_rows != expected_length:
        raise ValueError(
            f"episode {episode_index}: source has {table.num_rows} rows, "
            f"official metadata says {expected_length}"
        )
    episode_values = table["episode_index"].to_pylist()
    if set(episode_values) != {episode_index}:
        raise ValueError(f"episode {episode_index}: inconsistent episode_index column")

    frame_index = table["frame_index"].combine_chunks()
    expected_frames = list(range(expected_length))
    if frame_index.to_pylist() != expected_frames:
        raise ValueError(f"episode {episode_index}: non-contiguous frame_index")

    task_values = pa.array([task_index] * expected_length, type=pa.int64())
    table = table.set_column(
        table.schema.get_field_index("task_index"), "task_index", task_values
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, dst)
    return episode_index, expected_length


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError("output must be a sibling of source, not source or its child")

    source_info = json.loads((source / "meta/info.json").read_text())
    chunk_size = int(source_info["chunks_size"])

    tasks_table = pq.read_table(args.official_tasks)
    task_indices = tasks_table["task_index"].to_pylist()
    task_texts = tasks_table["task"].to_pylist()
    task_by_text = dict(zip(task_texts, task_indices, strict=True))
    if len(task_by_text) != len(task_texts):
        raise ValueError("official task text is not unique")

    episodes_table = pq.read_table(args.official_episodes)
    episode_rows = episodes_table.to_pylist()
    if args.max_episodes is not None:
        episode_rows = episode_rows[: args.max_episodes]

    episode_specs: list[tuple[int, int, int, str, dict[str, Any]]] = []
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        tasks = row["tasks"]
        if len(tasks) != 1:
            raise ValueError(f"episode {episode_index}: expected one task, got {tasks}")
        task_text = tasks[0]
        task_index = int(task_by_text[task_text])
        length = int(row["length"])
        episode_specs.append((episode_index, task_index, length, task_text, row))

    source_stats_by_episode: dict[int, dict[str, dict[str, Any]]] = {}
    with (source / "meta/episodes_stats.jsonl").open() as stats_file:
        for line in stats_file:
            item = json.loads(line)
            source_stats_by_episode[int(item["episode_index"])] = item["stats"]

    output.mkdir(parents=True, exist_ok=True)
    video_link = output / "videos"
    if not video_link.exists():
        video_link.symlink_to(source / "videos", target_is_directory=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                rewrite_episode,
                source,
                output,
                chunk_size,
                episode_index,
                task_index,
                length,
            )
            for episode_index, task_index, length, _, _ in episode_specs
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            future.result()
            if completed % 1000 == 0 or completed == len(futures):
                print(f"rewritten {completed}/{len(futures)} episodes", flush=True)

    info = source_info.copy()
    info["total_episodes"] = len(episode_specs)
    info["total_frames"] = sum(spec[2] for spec in episode_specs)
    info["total_tasks"] = len(task_texts)
    info["total_videos"] = len(episode_specs) * 3
    info["total_chunks"] = math.ceil(len(episode_specs) / chunk_size)
    info["splits"] = {"train": f"0:{len(episode_specs)}"}

    meta = output / "meta"
    atomic_text(meta / "info.json", json.dumps(info, indent=2) + "\n")
    atomic_text(
        meta / "tasks.jsonl",
        "".join(
            json.dumps({"task_index": index, "task": text}) + "\n"
            for index, text in zip(task_indices, task_texts, strict=True)
        ),
    )
    atomic_text(
        meta / "episodes.jsonl",
        "".join(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "tasks": [task_text],
                    "length": length,
                }
            )
            + "\n"
            for episode_index, _, length, task_text, _ in episode_specs
        ),
    )
    atomic_text(
        meta / "episodes_stats.jsonl",
        "".join(
            json.dumps(
                {
                    "episode_index": episode_index,
                    "stats": patched_stats(
                        row,
                        source_stats_by_episode[episode_index],
                        task_index,
                        length,
                    ),
                }
            )
            + "\n"
            for episode_index, task_index, length, _, row in episode_specs
        ),
    )
    manifest = {
        "builder": str(Path(__file__).resolve()),
        "source": str(source),
        "output": str(output),
        "episodes": len(episode_specs),
        "frames": info["total_frames"],
        "official_tasks": len(task_texts),
        "video_fps_preserved": info["fps"],
        "invariants": {
            "one_official_task_per_episode": True,
            "source_videos_symlinked_without_reencoding": video_link.is_symlink(),
            "source_timestamps_preserved": True,
        },
        "sha256": {
            "official_tasks": sha256(args.official_tasks),
            "official_episodes": sha256(args.official_episodes),
            "source_info": sha256(source / "meta/info.json"),
            "builder": sha256(Path(__file__)),
        },
    }
    atomic_text(
        meta / "official_prompt_repair_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": len(episode_specs),
                "frames": info["total_frames"],
                "tasks": info["total_tasks"],
                "fps": info["fps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

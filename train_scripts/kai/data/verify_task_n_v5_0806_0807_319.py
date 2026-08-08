#!/usr/bin/env python3
"""Verify and freeze the Task_N 08-06/08-07 319-episode dataset.

This verifier is intentionally separate from the builder: it can audit an
already-built 287/32 dataset without rewriting several gigabytes of videos.
It records content hashes for every admitted source and derived artifact while
keeping an unavailable TOS completion timestamp explicit rather than inferred.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_task_n_v5_0806_0807_319_joint14 as build  # noqa: E402
from norm_stats_from_dataset import compute_norm_stats  # noqa: E402


KAI0_ROOT = Path(os.environ.get("KAI0_ROOT", Path(__file__).resolve().parents[3] / "kai0"))
REPO_ROOT = KAI0_ROOT.parent
SOURCE_ROOT = KAI0_ROOT / "data" / "Task_N" / "base" / "v5"
TRAIN_ROOT = KAI0_ROOT / "data" / "Task_N" / "self_built" / "nail_v5_0806_0807_319_joint14_train"
VAL_ROOT = KAI0_ROOT / "data" / "Task_N" / "self_built" / "nail_v5_0806_0807_319_joint14_val"
DEFAULT_REPORT = REPO_ROOT / "docs" / "training" / "analysis" / "task_n_v5_0806_0807_319_freeze.json"

SOURCE_CAMERAS = (
    "observation.images.top_head",
    "observation.images.mid_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
OUTPUT_CAMERAS = build.base.CAMERAS
EXPECTED_ORPHANS = {
    "2026-08-06-v5": {154},
    "2026-08-07-v5": {298},
}
EXPECTED_SPLIT = {"train": (287, 196_203), "val": (32, 18_726)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "bytes": stat.st_size,
        "sha256": sha256(path),
    }


def hash_files(paths: list[Path], root: Path, workers: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda path: file_record(path, root), paths))
    return sorted(records, key=lambda item: item["path"])


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_video_path(date_root: Path, camera: str, episode_id: int) -> Path:
    path = date_root / "videos" / "chunk-000" / camera / f"episode_{episode_id:06d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def seek_midpoint(path: Path) -> str:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = int(stream.frames or 0)
        if frames <= 0:
            for index, _frame in enumerate(container.decode(video=0), 1):
                if index >= 2:
                    return "linear_decode"
            raise ValueError(f"video has no decodable frames: {path}")
        target = frames // 2
        if stream.duration is not None:
            target_ts = int(stream.duration * target / frames)
            container.seek(target_ts, stream=stream, any_frame=False, backward=True)
        try:
            next(container.decode(video=0))
            return "indexed_seek"
        except StopIteration:
            pass

    # A subset of source MP4 files has an incomplete seek index. Verify the
    # actual midpoint content by decoding from frame zero, matching the builder.
    with av.open(str(path)) as container:
        for index, _frame in enumerate(container.decode(video=0)):
            if index >= frames // 2:
                return "linear_fallback"
    raise ValueError(f"cannot decode midpoint: {path}")


def validate_split(root: Path, expected_count: int, expected_frames: int) -> tuple[list[dict], list[Path]]:
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    if (info["total_episodes"], info["total_frames"], len(episodes)) != (
        expected_count,
        expected_frames,
        expected_count,
    ):
        raise ValueError(f"bad split totals for {root}: {info['total_episodes']}/{info['total_frames']}/{len(episodes)}")
    if info["total_videos"] != expected_count * len(OUTPUT_CAMERAS):
        raise ValueError(f"bad video total for {root}: {info['total_videos']}")

    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    video_paths = sorted((root / "videos").rglob("*.mp4"))
    if len(parquet_paths) != expected_count or len(video_paths) != expected_count * len(OUTPUT_CAMERAS):
        raise ValueError(f"bad artifact counts for {root}: parquet={len(parquet_paths)} videos={len(video_paths)}")

    global_index = 0
    for episode_index, (meta, parquet) in enumerate(zip(episodes, parquet_paths, strict=True)):
        if meta["episode_index"] != episode_index or meta["length"] <= 0:
            raise ValueError(f"bad episode metadata at {root}:{episode_index}")
        frame = pd.read_parquet(parquet)
        state = np.stack(frame["observation.state"].to_numpy())
        action = np.stack(frame["action"].to_numpy())
        length = len(frame)
        if length != meta["length"] or state.shape != (length, 14) or action.shape != (length, 14):
            raise ValueError(f"bad shape/length at {parquet}: {state.shape}/{action.shape}/{length}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"NaN/Inf at {parquet}")
        if not np.array_equal(frame["episode_index"].to_numpy(), np.full(length, episode_index)):
            raise ValueError(f"non-contiguous episode_index at {parquet}")
        if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"non-contiguous frame_index at {parquet}")
        if not np.array_equal(frame["index"].to_numpy(), np.arange(global_index, global_index + length)):
            raise ValueError(f"non-contiguous global index at {parquet}")
        global_index += length
    if global_index != expected_frames:
        raise ValueError(f"bad accumulated frames for {root}: {global_index}")
    return episodes, parquet_paths + video_paths


def validate_source_mapping(split: list[dict], source_by_identity: dict[str, build.base.SourceEpisode], root: Path) -> None:
    for item in split:
        source = source_by_identity[item["source_identity"]]
        output = root / "data" / "chunk-000" / f"episode_{item['episode_index']:06d}.parquet"
        source_frame = pd.read_parquet(source.parquet, columns=["observation.state", "action"])
        output_frame = pd.read_parquet(output, columns=["observation.state", "action"])
        for column in ("observation.state", "action"):
            source_values = np.stack(source_frame[column].to_numpy())[:, :14].astype(np.float32)
            output_values = np.stack(output_frame[column].to_numpy())
            if not np.array_equal(source_values, output_values):
                raise ValueError(f"source mapping changed {column}: {item['source_identity']}")


def verify_norm_stats(train_root: Path) -> str:
    expected_path = train_root / "norm_stats.json"
    if not expected_path.is_file():
        raise FileNotFoundError(expected_path)
    with tempfile.TemporaryDirectory(prefix="task_n_319_norm_") as temp:
        temp_root = Path(temp)
        data_root = temp_root / "data" / "chunk-000"
        data_root.mkdir(parents=True)
        for parquet in sorted((train_root / "data").rglob("*.parquet")):
            os.symlink(parquet.resolve(), data_root / parquet.name)
        compute_norm_stats(str(temp_root), action_dim=32)
        actual = json.loads((temp_root / "norm_stats.json").read_text())
    expected = json.loads(expected_path.read_text())
    if actual.keys() != expected.keys():
        raise ValueError("norm_stats top-level schema mismatch")
    for feature, expected_stats in expected["norm_stats"].items():
        for key, expected_values in expected_stats.items():
            np.testing.assert_allclose(
                np.asarray(actual["norm_stats"][feature][key]),
                np.asarray(expected_values),
                rtol=0,
                atol=1e-12,
                err_msg=f"norm_stats mismatch: {feature}.{key}",
            )
    return sha256(expected_path)


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hash-workers", type=int, default=16)
    parser.add_argument("--seek-samples", type=int, default=16)
    parser.add_argument("--tos-sync-completed-at")
    parser.add_argument("--skip-norm-recompute", action="store_true")
    args = parser.parse_args()
    if args.tos_sync_completed_at:
        datetime.fromisoformat(args.tos_sync_completed_at.replace("Z", "+00:00"))

    episodes = build.discover()
    train_sources, val_sources, allocation = build.allocate_val(episodes)
    source_by_identity = {episode.identity: episode for episode in episodes}
    source_ids = set(source_by_identity)
    train_ids = {episode.identity for episode in train_sources}
    val_ids = {episode.identity for episode in val_sources}
    if len(episodes) != 319 or train_ids & val_ids or train_ids | val_ids != source_ids:
        raise ValueError("source/split identity invariant failed")

    metadata_records = []
    source_files: list[Path] = []
    source_entries = []
    observed_mtimes = []
    for date in build.DATES:
        date_root = SOURCE_ROOT / date
        metadata = date_root / "meta" / "by_station" / "ipc01" / "episodes.jsonl"
        metadata_records.append(file_record(metadata, SOURCE_ROOT))
        source_files.append(date_root / "meta" / "info.json")
        admitted_ids = {episode.source_episode_id for episode in episodes if episode.date == date}
        disk_ids = {
            int(path.stem.removeprefix("episode_"))
            for path in (date_root / "data" / "chunk-000").glob("episode_*.parquet")
        }
        orphan_ids = disk_ids - admitted_ids
        if orphan_ids != EXPECTED_ORPHANS[date]:
            raise ValueError(f"unexpected metadata-external parquets for {date}: {sorted(orphan_ids)}")
    for episode in episodes:
        all_videos = [source_video_path(episode.source_root, camera, episode.source_episode_id) for camera in SOURCE_CAMERAS]
        files = [episode.parquet, *all_videos]
        source_files.extend(files)
        observed_mtimes.extend(path.stat().st_mtime for path in files)
        source_entries.append(
            {
                "identity": episode.identity,
                "frames": int(episode.source_meta["length"]),
                "created_at": episode.created_at,
                "split": "train" if episode.identity in train_ids else "val",
                "files": [str(path.relative_to(SOURCE_ROOT)) for path in files],
            }
        )

    train_meta, train_artifacts = validate_split(TRAIN_ROOT, *EXPECTED_SPLIT["train"])
    val_meta, val_artifacts = validate_split(VAL_ROOT, *EXPECTED_SPLIT["val"])
    if {item["source_identity"] for item in train_meta} != train_ids:
        raise ValueError("derived train identities differ from frozen split")
    if {item["source_identity"] for item in val_meta} != val_ids:
        raise ValueError("derived val identities differ from frozen split")
    validate_source_mapping(train_meta, source_by_identity, TRAIN_ROOT)
    validate_source_mapping(val_meta, source_by_identity, VAL_ROOT)

    rng = random.Random(42)
    sampled = rng.sample(episodes, min(args.seek_samples, len(episodes)))
    seeked_source = []
    seeked_output = []
    output_location = {item["source_identity"]: (TRAIN_ROOT, item) for item in train_meta}
    output_location.update({item["source_identity"]: (VAL_ROOT, item) for item in val_meta})
    for episode in sampled:
        for camera in OUTPUT_CAMERAS:
            source_video = source_video_path(episode.source_root, camera, episode.source_episode_id)
            source_mode = seek_midpoint(source_video)
            seeked_source.append(
                {"path": str(source_video.relative_to(SOURCE_ROOT)), "mode": source_mode}
            )
            split_root, item = output_location[episode.identity]
            output_video = split_root / "videos" / "chunk-000" / camera / f"episode_{item['episode_index']:06d}.mp4"
            output_mode = seek_midpoint(output_video)
            seeked_output.append(
                {"path": str(output_video.relative_to(KAI0_ROOT)), "mode": output_mode}
            )

    norm_sha = sha256(TRAIN_ROOT / "norm_stats.json")
    if not args.skip_norm_recompute:
        norm_sha = verify_norm_stats(TRAIN_ROOT)

    source_hashes = hash_files(sorted(set(source_files)), SOURCE_ROOT, args.hash_workers)
    derived_files = train_artifacts + val_artifacts
    for root in (TRAIN_ROOT, VAL_ROOT):
        derived_files.extend(sorted((root / "meta").glob("*")))
    derived_files.append(TRAIN_ROOT / "norm_stats.json")
    derived_hashes = hash_files(
        sorted(set(derived_files)),
        KAI0_ROOT / "data" / "Task_N" / "self_built",
        args.hash_workers,
    )

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "complete": args.tos_sync_completed_at is not None,
        "incomplete_reasons": [] if args.tos_sync_completed_at else ["authoritative TOS sync completion time unavailable"],
        "source": {
            "root": str(SOURCE_ROOT),
            "episodes": 319,
            "frames": 214_929,
            "metadata": metadata_records,
            "tos_sync_completed_at": args.tos_sync_completed_at,
            "observed_file_mtime_window": {
                "first": datetime.fromtimestamp(min(observed_mtimes), timezone.utc).isoformat().replace("+00:00", "Z"),
                "last": datetime.fromtimestamp(max(observed_mtimes), timezone.utc).isoformat().replace("+00:00", "Z"),
                "note": "Local filesystem evidence only; not a substitute for the TOS completion timestamp.",
            },
            "excluded_orphans": [
                {"date": date, "episode_id": episode_id, "reason": "parquet exists but authoritative metadata does not reference it"}
                for date, ids in EXPECTED_ORPHANS.items()
                for episode_id in sorted(ids)
            ],
            "entries": source_entries,
            "file_hashes": source_hashes,
        },
        "split": {
            "strategy": "fixed tail holdout within date; 9/23 val allocation",
            "allocation": allocation,
            "train_episodes": 287,
            "train_frames": 196_203,
            "val_episodes": 32,
            "val_frames": 18_726,
            "identity_overlap": 0,
        },
        "derived": {
            "train_root": str(TRAIN_ROOT),
            "val_root": str(VAL_ROOT),
            "state_dim": 14,
            "action_dim": 14,
            "cameras": list(OUTPUT_CAMERAS),
            "source_to_output_values_exact": True,
            "norm_stats_train_only_recomputed": not args.skip_norm_recompute,
            "norm_stats_sha256": norm_sha,
            "file_hashes": derived_hashes,
        },
        "video_seek": {
            "seed": 42,
            "sampled_episodes": len(sampled),
            "source_files": seeked_source,
            "derived_files": seeked_output,
        },
        "code": {
            "git_head": git_output("rev-parse", "HEAD"),
            "git_status_short": git_output("status", "--short"),
            "builder_sha256": sha256(HERE / "build_task_n_v5_0806_0807_319_joint14.py"),
            "base_builder_sha256": sha256(HERE / "build_task_n_v5_272_joint14.py"),
            "verifier_sha256": sha256(Path(__file__)),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temp_report.write_text(json.dumps(report, indent=2) + "\n")
    temp_report.replace(args.report)
    print(
        "TASK_N_319_VERIFY_OK",
        f"complete={report['complete']}",
        f"source_files={len(source_hashes)}",
        f"derived_files={len(derived_hashes)}",
        f"report={args.report}",
    )


if __name__ == "__main__":
    main()

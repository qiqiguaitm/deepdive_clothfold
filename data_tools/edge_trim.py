#!/usr/bin/env python3
"""Safely trim only the leading and trailing idle runs of station datasets.

The source leaf is never edited in place.  ``--apply`` builds a fully validated
staging leaf, renames the source to a timestamped backup, then atomically renames
the staging leaf into place.  Interior pauses are always preserved.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FPS = 30
ARM_DIMS = list(range(0, 6)) + list(range(7, 13))
GRIP_DIMS = [6, 13]
ARM_THR = 3e-3
GRIP_THR = 0.02
MOTION_WIN = 10
FRONT_MARGIN = 15
TAIL_CAP = 15
EP_RE = re.compile(r"episode_(\d{6})\.parquet$")
CHUNK_RE = re.compile(r"^(\d+)(\..+)$")


@dataclass(frozen=True)
class TrimPlan:
    episode_id: int
    original_frames: int
    start: int
    end: int
    kept_frames: int
    front_removed: int
    tail_removed: int


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim station dataset edges through validated staging + atomic swap",
    )
    parser.add_argument("--leaf", action="append", required=True, type=Path)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--encoder", default="libx264")
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=18)
    return parser.parse_args()


def _action14(table: pa.Table) -> np.ndarray:
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if action.ndim != 2 or action.shape[1] < 14:
        raise ValueError(f"action must be [T,>=14], got {action.shape}")
    return action[:, :14]


def _motion_onset(action: np.ndarray) -> int:
    """Return the first diff index of 10 sustained arm-motion transitions."""
    if len(action) <= 1:
        return len(action)
    delta = np.abs(np.diff(action[:, ARM_DIMS], axis=0)).mean(axis=1)
    run = 0
    for idx, moving in enumerate(delta > ARM_THR):
        run = run + 1 if moving else 0
        if run >= MOTION_WIN:
            return idx - MOTION_WIN + 1
    return len(action)


def _tail_keep_end(action: np.ndarray) -> int:
    """Cap only the final static run; never inspect or thin an interior pause."""
    total = len(action)
    if total <= 1:
        return total
    d_arm = np.abs(np.diff(action[:, ARM_DIMS], axis=0)).mean(axis=1)
    d_grip = np.abs(np.diff(action[:, GRIP_DIMS], axis=0)).max(axis=1)
    active = np.concatenate([[True], (d_arm > ARM_THR) | (d_grip > GRIP_THR)])
    tail = 0
    for value in active[::-1]:
        if value:
            break
        tail += 1
    return total if tail <= TAIL_CAP else total - (tail - TAIL_CAP)


def _plan(parquet: Path) -> TrimPlan:
    table = pq.read_table(parquet, columns=["action"])
    action = _action14(table)
    total = len(action)
    start = max(0, _motion_onset(action) - FRONT_MARGIN)
    end = _tail_keep_end(action)
    if end <= start:
        raise ValueError(f"edge trim would empty {parquet}: start={start}, end={end}")
    ep_match = EP_RE.search(parquet.name)
    if not ep_match:
        raise ValueError(f"invalid parquet name: {parquet}")
    return TrimPlan(
        episode_id=int(ep_match.group(1)),
        original_frames=total,
        start=start,
        end=end,
        kept_frames=end - start,
        front_removed=start,
        tail_removed=total - end,
    )


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise ValueError(f"missing required parquet column: {name}")
    return table.set_column(idx, name, values)


def _trim_parquet(src: Path, dst: Path, plan: TrimPlan) -> None:
    table = pq.read_table(src).slice(plan.start, plan.kept_frames)
    n = plan.kept_frames
    table = _replace_column(table, "frame_index", pa.array(np.arange(n), type=pa.int64()))
    table = _replace_column(table, "index", pa.array(np.arange(n), type=pa.int64()))
    table = _replace_column(
        table, "timestamp", pa.array(np.arange(n, dtype=np.float32) / FPS, type=pa.float32()),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


def _trim_video(
    src: Path,
    dst: Path,
    plan: TrimPlan,
    encoder: str,
    preset: str,
    crf: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    decoded = 0
    written = 0
    with av.open(str(src)) as source, av.open(str(dst), mode="w") as output:
        input_stream = source.streams.video[0]
        input_stream.thread_type = "AUTO"
        output_stream = output.add_stream(encoder, rate=FPS)
        output_stream.width = input_stream.codec_context.width
        output_stream.height = input_stream.codec_context.height
        output_stream.pix_fmt = "yuv420p"
        if encoder == "libx264":
            output_stream.options = {
                "crf": str(crf), "preset": preset, "threads": "2",
            }
        for idx, frame in enumerate(source.decode(video=0)):
            decoded += 1
            if idx < plan.start:
                continue
            if idx >= plan.end:
                break
            trimmed = frame.reformat(
                width=output_stream.width,
                height=output_stream.height,
                format="yuv420p",
            )
            trimmed.pts = written
            trimmed.time_base = Fraction(1, FPS)
            for packet in output_stream.encode(trimmed):
                output.mux(packet)
            written += 1
        for packet in output_stream.encode():
            output.mux(packet)
    if decoded < plan.end or written != plan.kept_frames:
        raise ValueError(
            f"video source mismatch {src}: decoded={decoded}, written={written}, "
            f"expected_end={plan.end}, expected_written={plan.kept_frames}"
        )


def _trim_depth_zip(src: Path, dst: Path, plan: TrimPlan) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zin:
        names = set(zin.namelist())
        if ".zarray" not in names:
            raise ValueError(f"depth zip missing .zarray: {src}")
        zarray = json.loads(zin.read(".zarray"))
        if list(zarray.get("chunks", []))[:1] != [1]:
            raise ValueError(f"depth chunks are not frame-addressable: {src}")
        if int(zarray["shape"][0]) != plan.original_frames:
            raise ValueError(
                f"depth/parquet length mismatch {src}: {zarray['shape'][0]} != {plan.original_frames}"
            )
        zarray["shape"][0] = plan.kept_frames
        written_chunks = 0
        with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
            for info in zin.infolist():
                name = info.filename
                if name == ".zarray":
                    payload = json.dumps(zarray, separators=(",", ":")).encode()
                    zout.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
                    continue
                match = CHUNK_RE.match(name)
                if match:
                    old_idx = int(match.group(1))
                    if plan.start <= old_idx < plan.end:
                        new_name = f"{old_idx - plan.start}{match.group(2)}"
                        zout.writestr(new_name, zin.read(name), compress_type=zipfile.ZIP_STORED)
                        written_chunks += 1
                    continue
                zout.writestr(name, zin.read(name), compress_type=zipfile.ZIP_STORED)
    if written_chunks != plan.kept_frames:
        raise ValueError(
            f"depth chunk count mismatch {src}: {written_chunks} != {plan.kept_frames}"
        )
    with zipfile.ZipFile(dst, "r") as check:
        if check.testzip() is not None:
            raise ValueError(f"corrupt generated depth zip: {dst}")


def _video_frame_info(path: Path) -> tuple[int, int | None]:
    count = 0
    first_pts: int | None = None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if first_pts is None:
                first_pts = frame.pts
            count += 1
    return count, first_pts


def _validate_parquet(path: Path, plan: TrimPlan) -> None:
    table = pq.read_table(path, columns=["action", "frame_index", "index", "timestamp"])
    if table.num_rows != plan.kept_frames:
        raise ValueError(f"parquet rows mismatch: {path}")
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    index = np.asarray(table["index"].to_pylist(), dtype=np.int64)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    if not np.array_equal(frame_index, np.arange(plan.kept_frames)):
        raise ValueError(f"non-contiguous frame_index: {path}")
    if not np.array_equal(index, np.arange(plan.kept_frames)):
        raise ValueError(f"non-contiguous index: {path}")
    if not np.allclose(timestamp, np.arange(plan.kept_frames) / FPS, atol=2e-6):
        raise ValueError(f"timestamp mismatch: {path}")
    action = _action14(table)
    if max(0, _motion_onset(action) - FRONT_MARGIN) != 0:
        raise ValueError(f"leading idle remains after trim: {path}")
    if _tail_keep_end(action) != len(action):
        raise ValueError(f"trailing idle remains after trim: {path}")


def _process_episode(
    leaf: Path,
    staging: Path,
    chunk_dir: str,
    parquet: Path,
    plan: TrimPlan,
    video_features: list[str],
    depth_features: list[str],
    encoder: str,
    preset: str,
    crf: int,
) -> TrimPlan:
    ep_name = f"episode_{plan.episode_id:06d}"
    out_parquet = staging / "data" / chunk_dir / f"{ep_name}.parquet"
    _trim_parquet(parquet, out_parquet, plan)
    _validate_parquet(out_parquet, plan)
    for feature in video_features:
        src = leaf / "videos" / chunk_dir / feature / f"{ep_name}.mp4"
        dst = staging / "videos" / chunk_dir / feature / f"{ep_name}.mp4"
        if not src.is_file():
            raise FileNotFoundError(src)
        _trim_video(src, dst, plan, encoder, preset, crf)
        frames, first_pts = _video_frame_info(dst)
        if frames != plan.kept_frames or first_pts != 0:
            raise ValueError(
                f"video validation failed {dst}: frames={frames}, first_pts={first_pts}, "
                f"expected={plan.kept_frames}/0"
            )
    for feature in depth_features:
        src = leaf / "videos" / chunk_dir / feature / f"{ep_name}.zarr.zip"
        dst = staging / "videos" / chunk_dir / feature / f"{ep_name}.zarr.zip"
        if not src.is_file():
            raise FileNotFoundError(src)
        _trim_depth_zip(src, dst, plan)
    return plan


def _copy_and_update_meta(
    leaf: Path,
    staging: Path,
    chunk_dir: str,
    plans: dict[int, TrimPlan],
    video_feature_count: int,
) -> None:
    src_meta = leaf / "meta"
    dst_meta = staging / "meta"
    if not src_meta.is_dir():
        raise FileNotFoundError(src_meta)
    shutil.copytree(src_meta, dst_meta)
    episodes_path = dst_meta / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    row_ids = {
        int(row.get("episode_id", row.get("episode_index")))
        for row in rows
        if str(row.get("chunk", chunk_dir)) == chunk_dir
    }
    if row_ids != set(plans):
        raise ValueError(
            f"metadata/parquet ids differ: metadata={sorted(row_ids)}, parquet={sorted(plans)}"
        )
    for row in rows:
        if str(row.get("chunk", chunk_dir)) != chunk_dir:
            raise ValueError("mixed chunks in one station leaf are not supported by this tool")
        ep = int(row.get("episode_id", row.get("episode_index")))
        plan = plans[ep]
        row["length"] = plan.kept_frames
        row["duration_s"] = round(plan.kept_frames / FPS, 3)
        row["edge_trim"] = {
            "front_removed": plan.front_removed,
            "tail_removed": plan.tail_removed,
            "interior_pauses_preserved": True,
            "policy": "arm_sustained_front_margin15+arm_gripper_tail_cap15",
        }
    episodes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    info_path = dst_meta / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = len(rows)
    info["total_frames"] = sum(int(row["length"]) for row in rows)
    info["total_videos"] = len(rows) * video_feature_count
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(rows)}"}
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    report = {
        "format": "kai0.edge-trim.v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_leaf": str(leaf),
        "chunk": chunk_dir,
        "interior_pauses_preserved": True,
        "total_original_frames": sum(plan.original_frames for plan in plans.values()),
        "total_kept_frames": sum(plan.kept_frames for plan in plans.values()),
        "total_front_removed": sum(plan.front_removed for plan in plans.values()),
        "total_tail_removed": sum(plan.tail_removed for plan in plans.values()),
        "episodes": [asdict(plans[ep]) for ep in sorted(plans)],
    }
    (dst_meta / "edge_trim_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _feature_dirs(leaf: Path, chunk_dir: str, prefix: str) -> list[str]:
    base = leaf / "videos" / chunk_dir
    return sorted(path.name for path in base.glob(f"{prefix}*") if path.is_dir())


def _trim_leaf(args: argparse.Namespace, leaf: Path) -> dict:
    leaf = leaf.resolve()
    chunk_dir = f"chunk-{args.chunk:03d}"
    parquet_dir = leaf / "data" / chunk_dir
    parquets = sorted(parquet_dir.glob("episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquets: {parquet_dir}")
    plans: dict[int, TrimPlan] = {}
    for path in parquets:
        plan = _plan(path)
        plans[plan.episode_id] = plan
    summary = {
        "leaf": str(leaf),
        "chunk": chunk_dir,
        "episodes": len(plans),
        "original_frames": sum(plan.original_frames for plan in plans.values()),
        "kept_frames": sum(plan.kept_frames for plan in plans.values()),
        "front_removed": sum(plan.front_removed for plan in plans.values()),
        "tail_removed": sum(plan.tail_removed for plan in plans.values()),
    }
    summary["removed_percent"] = round(
        100 * (summary["original_frames"] - summary["kept_frames"])
        / summary["original_frames"],
        3,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not args.apply:
        return summary

    staging = leaf.parent / f".{leaf.name}.edge_trim_staging"
    backup = leaf.parent / f"{leaf.name}.pre_edge_trim_{args.backup_tag}"
    if staging.exists() or backup.exists():
        raise FileExistsError(f"refusing existing staging/backup: {staging} / {backup}")
    video_features = _feature_dirs(leaf, chunk_dir, "observation.images.")
    depth_features = _feature_dirs(leaf, chunk_dir, "observation.depth.")
    if not video_features:
        raise ValueError(f"no RGB video feature directories: {leaf}")
    staging.mkdir(parents=True)
    try:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="edge-trim") as pool:
            for parquet in parquets:
                plan = plans[int(EP_RE.search(parquet.name).group(1))]
                future = pool.submit(
                    _process_episode,
                    leaf,
                    staging,
                    chunk_dir,
                    parquet,
                    plan,
                    video_features,
                    depth_features,
                    args.encoder,
                    args.preset,
                    args.crf,
                )
                futures[future] = plan
            done = 0
            for future in as_completed(futures):
                plan = future.result()
                done += 1
                print(
                    f"progress={done}/{len(futures)} ep={plan.episode_id:06d} "
                    f"frames={plan.original_frames}->{plan.kept_frames}",
                    flush=True,
                )
        _copy_and_update_meta(
            leaf, staging, chunk_dir, plans, video_feature_count=len(video_features),
        )
        # Preserve any non-standard root entries without touching generated trees.
        for entry in leaf.iterdir():
            if entry.name in {"data", "videos", "meta"}:
                continue
            target = staging / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
        os.rename(leaf, backup)
        try:
            os.rename(staging, leaf)
        except Exception:
            os.rename(backup, leaf)
            raise
    except Exception:
        # Keep staging for forensic inspection/resume diagnosis; source is untouched
        # unless the final atomic swap succeeded.
        raise
    summary["backup"] = str(backup)
    summary["report"] = str(leaf / "meta" / "edge_trim_report.json")
    print(json.dumps({"complete": summary}, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    args = _args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    for leaf in args.leaf:
        _trim_leaf(args, leaf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

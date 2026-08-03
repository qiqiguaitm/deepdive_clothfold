#!/usr/bin/env python3
"""
dagger_trim.py — 物理裁掉 chunk-001 拼接 episode 中的非 ideal 帧。

删除 parquet 中 dagger_frame_class ∈ {2, 3, 4} 的行 (preintv/hesitation/stationary_tail),
并用 ffmpeg select 过滤器重编码视频 (移除对应帧)。

用法:
  # Dry-run 一个日期
  python trim_stitched_episodes.py --date 2026-06-24 --dry-run

  # 执行一个日期
  python trim_stitched_episodes.py --date 2026-06-24

  # 全量
  python trim_stitched_episodes.py --all
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("KAI0_DATA_ROOT", "/data1/DATA_IMP/KAI0"))
CAM_DIRS = ["observation.images.top_head", "observation.images.mid_head",
            "observation.images.hand_left", "observation.images.hand_right"]

# 要裁掉的 class
TRIM_CLASSES = {2, 3, 4}  # preintv, hesitation, stationary_tail


def trim_parquet(pq_path: str) -> tuple:
    """从 parquet 删除 TRIM_CLASSES 行, 覆盖写回.

    Returns: (orig_frames, kept_frames, trimmed_per_class)
    """
    t = pq.read_table(pq_path)
    classes = t.column("dagger_frame_class").to_pylist()
    n = len(classes)

    keep_mask = np.array([c not in TRIM_CLASSES for c in classes])
    keep_indices = np.where(keep_mask)[0]
    n_keep = len(keep_indices)

    if n_keep == n:
        return (n, n, {})

    # 统计
    trimmed = {}
    for c in TRIM_CLASSES:
        cnt = classes.count(c)
        if cnt > 0:
            trimmed[c] = cnt

    # 构建 trimmed table
    new_table = t.filter(pa.array(keep_mask))
    # 更新 frame_index, index, timestamp
    new_frame_idx = pa.array(list(range(n_keep)), type=pa.int64())
    new_index = pa.array(list(range(n_keep)), type=pa.int64())
    new_ts = pa.array(np.arange(n_keep, dtype=np.float32) / 30.0, type=pa.float32())

    cols = {"frame_index": new_frame_idx, "index": new_index, "timestamp": new_ts}
    for col_name in new_table.column_names:
        if col_name not in cols:
            cols[col_name] = new_table.column(col_name)

    final_table = pa.table(cols)
    pq.write_table(final_table, pq_path)

    return (n, n_keep, trimmed, keep_indices)


def _keep_ranges(keep_indices: np.ndarray) -> list[tuple[int, int]]:
    """将 keep 帧索引压缩为连续区间 [(start, end), ...]."""
    if len(keep_indices) == 0:
        return []
    ranges = []
    start = keep_indices[0]
    end = start
    for i in range(1, len(keep_indices)):
        if keep_indices[i] == end + 1:
            end = keep_indices[i]
        else:
            ranges.append((int(start), int(end)))
            start = keep_indices[i]
            end = start
    ranges.append((int(start), int(end)))
    return ranges


def trim_video(src_path: str, dst_path: str, keep_indices: np.ndarray) -> None:
    """用 ffmpeg select 过滤器裁剪视频帧.

    keep_indices: 要保留的帧索引 (0-based).
    用 between(n,start,end) 表达连续区间, 避免表达式过长.
    """
    n_keep = len(keep_indices)
    if n_keep == 0:
        return

    ranges = _keep_ranges(keep_indices)
    # 构建 select 表达式: between(n,0,100)+between(n,150,300)+...
    parts = [f"between(n,{s},{e})" for s, e in ranges]
    select_expr = "+".join(parts)

    # 如果表达式仍然太长 (>32KB), 写入文件用 -filter_script
    use_script = len(select_expr) > 30000
    script_file = None

    if use_script:
        script_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False).name
        with open(script_file, "w") as f:
            f.write(f"select='{select_expr}',setpts=N/FRAME_RATE/TB\n")

    try:
        if use_script:
            cmd = [
                "ffmpeg", "-y",
                "-i", src_path,
                "-filter_script:v", script_file,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                str(dst_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", src_path,
                "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                str(dst_path),
            ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    finally:
        if script_file and os.path.exists(script_file):
            os.unlink(script_file)


def trim_episode(pq_path: str, date_dir: str, ep_id: int, dry_run: bool = False) -> dict:
    """裁单个 episode 的 parquet + 视频."""
    if dry_run:
        t = pq.read_table(pq_path)
        classes = t.column("dagger_frame_class").to_pylist()
        trimmed = {c: classes.count(c) for c in TRIM_CLASSES if classes.count(c) > 0}
        return {"n_orig": len(classes), "n_keep": len(classes) - sum(trimmed.values()),
                "trimmed": trimmed}

    result = trim_parquet(pq_path)
    n_orig, n_keep, trimmed, keep_indices = result

    if n_keep == n_orig:
        return {"n_orig": n_orig, "n_keep": n_keep, "trimmed": {}}

    # 裁视频
    for cam in CAM_DIRS:
        vid_src = f"{date_dir}/videos/chunk-001/{cam}/episode_{ep_id:06d}.mp4"
        if not os.path.exists(vid_src):
            continue
        # 写到临时文件再换入
        vid_tmp = f"{vid_src}.trimtmp.mp4"
        try:
            trim_video(vid_src, vid_tmp, keep_indices)
            os.replace(vid_tmp, vid_src)
        except Exception as e:
            print(f"  [WARN] 视频裁切失败 {cam}: {e}")
            if os.path.exists(vid_tmp):
                os.unlink(vid_tmp)

    return {"n_orig": n_orig, "n_keep": n_keep, "trimmed": trimmed}


def trim_date(date: str, dry_run: bool = False) -> dict:
    """裁一个日期下所有 chunk-001 episode."""
    date_dir = DATA_ROOT / "Task_A" / "dagger" / "v4" / date
    pq_dir = date_dir / "data" / "chunk-001"
    if not pq_dir.exists():
        print(f"[{date}] 没有 chunk-001, 跳过")
        return {"date": date, "n_episodes": 0}

    parquets = sorted(pq_dir.glob("episode_*.parquet"))
    if not parquets:
        return {"date": date, "n_episodes": 0}

    total_orig = 0
    total_keep = 0
    total_trimmed = {c: 0 for c in TRIM_CLASSES}

    for pq_file in parquets:
        ep_id = int(pq_file.stem.replace("episode_", ""))
        r = trim_episode(str(pq_file), str(date_dir), ep_id, dry_run)
        total_orig += r["n_orig"]
        total_keep += r["n_keep"]
        for c, cnt in r["trimmed"].items():
            total_trimmed[c] += cnt

    # 更新 episodes_stitched.jsonl
    if not dry_run:
        meta_path = date_dir / "meta" / "episodes_stitched.jsonl"
        if meta_path.exists():
            updated = []
            for line in open(meta_path):
                e = json.loads(line)
                ep_id = e["episode_id"]
                pq_file = pq_dir / f"episode_{ep_id:06d}.parquet"
                if pq_file.exists():
                    t = pq.read_metadata(str(pq_file))
                    e["length"] = t.num_rows
                    e["duration_s"] = round(t.num_rows / 30.0, 3)
                    e["note"] = e.get("note", "") + " [trimmed]"
                updated.append(e)
            with open(meta_path, "w") as f:
                for e in updated:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

    class_names = {2: "preintv", 3: "hesitation", 4: "stationary"}
    print(f"[{date}] {len(parquets)} eps: {total_orig:,} → {total_keep:,} 帧 "
          f"(-{total_orig - total_keep:,}, {(total_orig-total_keep)/max(1,total_orig)*100:.1f}%)")
    for c in sorted(total_trimmed):
        if total_trimmed[c] > 0:
            print(f"  class {c} ({class_names[c]}): -{total_trimmed[c]:,} 帧")

    return {"date": date, "n_episodes": len(parquets),
            "orig": total_orig, "keep": total_keep, "trimmed": total_trimmed}


def main():
    parser = argparse.ArgumentParser(description="物理裁掉拼接 episode 中的非 ideal 帧")
    parser.add_argument("--date", type=str, help="指定日期")
    parser.add_argument("--all", action="store_true", help="所有日期")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.date and not args.all:
        parser.error("需要 --date 或 --all")

    if args.date:
        date = args.date if args.date.endswith("-v4") else f"{args.date}-v4"
        trim_date(date, args.dry_run)
    else:
        grand_orig = 0
        grand_keep = 0
        for d in sorted((DATA_ROOT / "Task_A" / "dagger" / "v4").glob("*-v4")):
            r = trim_date(d.name, args.dry_run)
            grand_orig += r.get("orig", 0)
            grand_keep += r.get("keep", 0)
        print(f"\n{'='*60}")
        print(f"全量: {grand_orig:,} → {grand_keep:,} 帧 (-{grand_orig-grand_keep:,})")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

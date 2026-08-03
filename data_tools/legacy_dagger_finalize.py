#!/usr/bin/env python3
"""
legacy_dagger_finalize.py — 历史拼接完成后执行（新静态检查请用 data_tools static）:
  1. 删除 >5min 的 episode
  2. 删除 review_episodes_gt2min.csv 中 if_delet=1 的 episode
  3. 14维逐维判定静止段, 生成 static_segments.csv
  4. 生成 core_s 统计 CSV

用法:
  python finalize_dagger_dataset.py
"""

import csv, json, os, glob
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

DATA = "/data1/DATA_IMP/KAI0/Task_A/dagger/v4"
LOG = "/data1/DATA_IMP/KAI0/Task_A/dagger/deleted_episodes_log.jsonl"
REVIEW_CSV = "/home/tim/Desktop/review_episodes_gt2min.csv"
FPS = 30
CAM_DIRS = ["observation.images.top_head", "observation.images.mid_head",
            "observation.images.hand_left", "observation.images.hand_right"]

# ── 收集待删除 ──
to_delete = set()  # (date, episode_name)

# 从 review CSV 读 if_delet=1
if os.path.exists(REVIEW_CSV):
    with open(REVIEW_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("if_delet", "").strip() == "1":
                date = row["date"]
                if not date.endswith("-v4"):
                    date = f"{date}-v4"
                to_delete.add((date, f"episode_{int(row['episode_id']):06d}"))

# 加上 >5min
for date_dir in sorted(glob.glob(f"{DATA}/*-v4")):
    date = os.path.basename(date_dir)
    pq_dir = f"{date_dir}/data/chunk-001"
    if not os.path.exists(pq_dir):
        continue
    for pq_file in glob.glob(f"{pq_dir}/episode_*.parquet"):
        t = pq.read_table(pq_file)
        dur = t.num_rows / FPS
        if dur > 300:
            to_delete.add((date, Path(pq_file).stem))

# ── Step 1: 删除 ──
deleted = []
for date_dir in sorted(glob.glob(f"{DATA}/*-v4")):
    date = os.path.basename(date_dir)
    meta_path = f"{date_dir}/meta/episodes_stitched.jsonl"
    pq_dir = f"{date_dir}/data/chunk-001"
    if not os.path.exists(meta_path) or not os.path.exists(pq_dir):
        continue

    episodes = [json.loads(l) for l in open(meta_path)]
    to_remove = set()
    for e in episodes:
        ename = f"episode_{e['episode_id']:06d}"
        if (date, ename) in to_delete:
            to_remove.add(e["episode_id"])
            deleted.append({
                "date": date, "episode_id": e["episode_id"],
                "duration_s": e["duration_s"], "length": e["length"],
                "reason": "最终排除", "segments": e.get("stitch_segments", []),
            })

    if not to_remove:
        continue

    for ep_id in to_remove:
        for fpath in [f"{pq_dir}/episode_{ep_id:06d}.parquet"] + \
                     [f"{date_dir}/videos/chunk-001/{cam}/episode_{ep_id:06d}.mp4" for cam in CAM_DIRS]:
            if os.path.exists(fpath): os.remove(fpath)

    kept = [e for e in episodes if e["episode_id"] not in to_remove]
    with open(meta_path, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[{date}] 删除 {len(to_remove)} eps, 保留 {len(kept)} eps")

if deleted:
    with open(LOG, "w" if not os.path.exists(LOG) else "a") as f:
        for d in deleted:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"删除 {len(deleted)} 个, 日志: {LOG}")

# ── Step 2: 14维逐维静止分析 ──
STAT_THR = 3e-3
MIN_STATIC = 60

static_rows = []
core_rows = []

for date_dir in sorted(glob.glob(f"{DATA}/*-v4")):
    date = os.path.basename(date_dir)
    pq_dir = f"{date_dir}/data/chunk-001"
    if not os.path.exists(pq_dir):
        continue
    for pq_file in sorted(glob.glob(f"{pq_dir}/episode_*.parquet")):
        t = pq.read_table(pq_file)
        n, classes = t.num_rows, t.column("dagger_frame_class").to_pylist()
        states = np.array([t.column("observation.state")[i].as_py() for i in range(n)], dtype=np.float64)

        # 14维逐维 velocity
        delta = np.abs(np.diff(states, axis=0))
        vel = np.zeros((n, 14)); vel[1:] = delta
        is_static = np.all(vel < STAT_THR, axis=1)

        # 静止段
        sr, in_run, rs = [], False, 0
        for i in range(n):
            if is_static[i] and not in_run: rs = i; in_run = True
            elif not is_static[i] and in_run: sr.append(i - rs); in_run = False
        if in_run: sr.append(n - rs)
        total_static = sum(sr)
        long_sr = [r for r in sr if r >= MIN_STATIC]

        ename = Path(pq_file).stem
        static_rows.append({
            "date": date, "episode": ename, "total_s": f"{n/FPS:.1f}",
            "static_s": f"{total_static/FPS:.1f}", "static_pct": f"{total_static/n*100:.1f}",
            "n_static_runs": len(sr), "n_long_gt2s": len(long_sr),
            "max_static_s": f"{max(sr)/FPS:.0f}" if sr else "0",
            "max_static_f": max(sr) if sr else 0,
            "runs_detail": " | ".join(f"{r}f({r/FPS:.0f}s)" for r in sorted(long_sr, reverse=True)[:10]),
        })

        # core 分析
        cr, in_run, rs = [], False, 0
        for i in range(n):
            if classes[i] == 1 and not in_run: rs = i; in_run = True
            elif classes[i] != 1 and in_run: cr.append(i - rs); in_run = False
        if in_run: cr.append(n - rs)
        total_core = sum(cr)
        core_rows.append({
            "date": date, "episode": ename, "total_s": f"{n/FPS:.1f}",
            "core_s": f"{total_core/FPS:.1f}", "pct": f"{total_core/n*100:.1f}",
            "n_core_gt2s": sum(1 for r in cr if r > 60),
        })

# ── Step 3: 写 CSV ──
OUT = "/data1/tim/workspace/deepdive_kai0/train_scripts/kai/data"
static_rows.sort(key=lambda r: -float(r["static_s"]))
core_rows.sort(key=lambda r: -float(r["core_s"]))

with open(f"{OUT}/static_segments.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date","episode","total_s","static_s","static_pct",
        "n_static_runs","n_long_gt2s","max_static_s","max_static_f","runs_detail"])
    w.writeheader(); w.writerows(static_rows)

with open(f"{OUT}/long_ideal_episodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["date","episode","total_s","core_s","pct","n_core_gt2s"])
    w.writeheader(); w.writerows(core_rows)

print(f"\nstatic_segments.csv — {len(static_rows)} 行")
print(f"long_ideal_episodes.csv — {len(core_rows)} 行")
print(f">2s 连续静止: {sum(1 for r in static_rows if int(r['n_long_gt2s'])>0)}")
print(f">5s 连续静止: {sum(1 for r in static_rows if int(r['max_static_f'])>150)}")

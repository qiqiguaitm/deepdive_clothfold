#!/usr/bin/env python3
"""Build A_v4_chunk001_dagger_crave_labeled: 387 base + 387 dagger with CRAVE stage_progress_gt.

Reads per-episode CRAVE polyline labels (npy), injects ``stage_progress_gt`` column into source
parquet, merges base (kai0_base sampled 387 ep) + dagger (chunk-001 full ep, 12 dates, no trim),
symlinks videos, re-indexes episodes 0..773, computes norm_stats.

Labels come from the CRAVE labeling pipeline (§1 of crave_value_dagger_validation_plan):
  lmvla/crave/temp/crave_ae_labels/chunk001_val/
    base/ep{ep_id}.npy       — kai0_base original episode_index (0-3054), 387 sampled
    dagger/ep{global_id}.npy  — global_id = date_hash*10000 + local_ep

Dagger global ID encoding (plan §6.2):
  date_hash = month*100 + day  (e.g. 06-16 → 616, 07-01 → 701)
  global_id // 10000 → date_hash, global_id % 10000 → local_ep
  Source: vis_dagger/v4/2026-{month:02d}-{day:02d}-v4/data/chunk-001/episode_{local_ep:06d}.parquet

Run (on gf3 where labels + source data live):
  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 \
  .venv/bin/python train_scripts/kai/data/build_chunk001_dagger_crave_labeled.py [--dry-run]

After build: discretize_advantage.py --data ... --advantage-source stage_progress_gt --discretion-type binary --top 0.3
"""
from __future__ import annotations

import argparse, json, os, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats  # noqa: E402

ROOT = Path(os.environ.get("KAI0_ROOT", "/vePFS/tim/workspace/deepdive_kai0/kai0"))
WORKSPACE = ROOT.parent  # lmvla/ lives at workspace root, not under kai0/
TASK_A = ROOT / "data" / "Task_A"
BASE_SRC = TASK_A / "kai0_base"                          # full 3055 ep; we sample 387
DAGGER_V4 = TASK_A / "vis_dagger" / "v4"                 # <date>/data/chunk-001/
LABEL_DIR = WORKSPACE / "lmvla" / "crave" / "temp" / "crave_ae_labels" / "chunk001_val"
OUT = TASK_A / "self_built" / os.environ.get("OUT_NAME", "A_v4_chunk001_dagger_crave_labeled")

CAMERAS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
FPS = 30
CHUNK = 0
PROMPT = "Flatten and fold the cloth."
# Standard LeRobot columns + stage_progress_gt (task_index gets filled by discretize step).
# 改动1: 保留逐帧 intervention / dagger_frame_class 标记(chunk-001 dagger 才有; base 无 → 下方填默认),
# 供 marker-based 打标(positive⟺人控)使用。见 dagger_launchpoint_trim_freeze_fix / velocity_gate 分析。
KEEP_COLS = [
    "observation.state", "action", "timestamp", "frame_index",
    "episode_index", "index", "task_index",
    "intervention", "dagger_frame_class",
]


def decode_dagger_global_id(global_id: int):
    """Return (date_dir, local_ep) from global_id = date_hash*10000 + local_ep."""
    dh = global_id // 10000
    local_ep = global_id % 10000
    month = dh // 100
    day = dh % 100
    date_dir = f"2026-{month:02d}-{day:02d}-v4"
    return date_dir, local_ep


def _find_pq(data_dir: Path, ep_id: int) -> Path | None:
    """Find a parquet in multi-chunk kai0_base (chunk-{ep//1000:03d})."""
    chunk = ep_id // 1000
    pq = data_dir / f"chunk-{chunk:03d}" / f"episode_{ep_id:06d}.parquet"
    return pq if pq.exists() else None


def discover_episodes():
    """Scan label directories, return [(group, ep_id, local_ep, label_npy_path, source_pq_path, src_date_dir)].

    For base: ep_id == local_ep (kai0_base original episode_index).
    For dagger: ep_id == global_id (date_hash*10000+local_ep), local_ep used for video lookup.
    """
    items = []

    # --- base ---
    base_label_dir = LABEL_DIR / "base"
    if base_label_dir.is_dir():
        for npy_path in sorted(base_label_dir.glob("ep*.npy")):
            ep_id = int(npy_path.stem.replace("ep", ""))
            pq_path = _find_pq(BASE_SRC / "data", ep_id)
            if pq_path:
                items.append(("base", ep_id, ep_id, npy_path, pq_path, None))
            else:
                print(f"  skip base ep{ep_id}: parquet 缺失 (all chunks)", flush=True)
    print(f"  base: {sum(1 for x in items if x[0]=='base')} ep discovered", flush=True)

    # --- dagger ---
    dagger_label_dir = LABEL_DIR / "dagger"
    if dagger_label_dir.is_dir():
        for npy_path in sorted(dagger_label_dir.glob("ep*.npy")):
            global_id = int(npy_path.stem.replace("ep", ""))
            date_dir, local_ep = decode_dagger_global_id(global_id)
            pq_path = (
                DAGGER_V4 / date_dir / "data" / "chunk-001"
                / f"episode_{local_ep:06d}.parquet"
            )
            if pq_path.exists():
                items.append(("dagger", global_id, local_ep, npy_path, pq_path, date_dir))
            else:
                print(f"  skip dagger {date_dir} ep{local_ep}: parquet 缺失 {pq_path}", flush=True)
    print(f"  dagger: {sum(1 for x in items if x[0]=='dagger')} ep discovered", flush=True)

    return items


def find_video(group: str, src_date_dir: str | None, ep_id: int, cam: str) -> Path:
    """Locate source video mp4.  base→kai0_base (multi-chunk), dagger→vis_dagger/v4/<date> (chunk-001).

    Tries both `observation.images.*` and bare camera names (new online recorder format).
    """
    if group == "base":
        root = BASE_SRC
        chunk_dir = f"chunk-{ep_id // 1000:03d}"
    else:
        root = DAGGER_V4 / src_date_dir
        chunk_dir = "chunk-001"

    for c in (cam, cam.replace("observation.images.", "")):
        p = root / "videos" / chunk_dir / c / f"episode_{ep_id:06d}.mp4"
        if p.exists() or p.is_symlink():
            return p
    raise FileNotFoundError(f"video 缺失: {group} {src_date_dir or 'base'} {cam} ep{ep_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    a = ap.parse_args()

    # --- discover ---
    items = discover_episodes()
    nb = sum(1 for x in items if x[0] == "base")
    nd = sum(1 for x in items if x[0] == "dagger")
    if nb == 0 and nd == 0:
        print("FATAL: 未发现任何 episode. 检查 LABEL_DIR 和源数据.", file=sys.stderr)
        sys.exit(1)
    print(f"总计: base={nb} + dagger={nd} = {len(items)} ep; KAI0_ROOT={ROOT}", flush=True)

    if a.dry_run:
        # Show a few examples
        for grp, ep_id, local_ep, npy_path, pq_path, src_date in items[:5]:
            label = np.load(npy_path)
            print(f"  {grp} global={ep_id} local={local_ep}: label shape={label.shape}, pq={pq_path.name}")
        print(f"  ... and {len(items)-5} more")
        print("dry-run: nothing written")
        return

    # --- output ---
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
    (OUT / "meta").mkdir()

    eps_meta, stats_out = [], []
    total_frames, new_ep = 0, 0
    skipped_label_mismatch = 0

    for grp, ep_id, local_ep, npy_path, pq_path, src_date in items:
        # Load label
        stage_gt = np.load(npy_path).astype(np.float32)  # [T]  0→1

        # Load source parquet
        df = pd.read_parquet(pq_path)
        df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

        # 改动1: base 源(kai0_base)无 intervention/dagger_frame_class 列 →
        # base = 人遥操专家示范: intervention=1(人在控制, A1 human 标注归 positive),
        # dagger_frame_class=5(CLASS_DEMO, Sirius 权重 w=1; 不能填 1=intv_core 否则被 A2/A3 上采样 2×)。
        # 保证 774 ep 输出 schema 一致(全含这两列 int64)。
        df["intervention"] = df["intervention"].astype("int64") if "intervention" in df.columns else np.int64(1)
        df["dagger_frame_class"] = (
            df["dagger_frame_class"].astype("int64") if "dagger_frame_class" in df.columns else np.int64(5)
        )

        n_label = len(stage_gt)
        n_pq = len(df)
        if n_label != n_pq:
            print(f"  WARNING: {grp} ep{ep_id} label length {n_label} != parquet {n_pq} — skipping",
                  flush=True)
            skipped_label_mismatch += 1
            continue

        # Inject stage_progress_gt
        df["stage_progress_gt"] = stage_gt

        # Reindex
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n_pq, dtype=np.int64)
        df["frame_index"] = np.arange(n_pq, dtype=np.int64)
        df["timestamp"] = (np.arange(n_pq, dtype=np.float32) / FPS)
        df["task_index"] = np.int64(0)  # placeholder, filled by discretize step

        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            OUT / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet",
        )

        # Symlink videos (use local_ep for video lookup)
        for cam in CAMERAS:
            sv = find_video(grp, src_date, local_ep, cam)
            dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(sv.resolve()), dv)

        eps_meta.append({
            "episode_index": new_ep,
            "tasks": [PROMPT],
            "length": n_pq,
            "src_ep": ep_id,
            "group": grp,
            "src_date": src_date,
        })
        stats_out.append({"episode_index": new_ep, "stats": per_episode_stats(df)})
        total_frames += n_pq
        new_ep += 1

    if skipped_label_mismatch:
        print(f"  ⚠ {skipped_label_mismatch} episodes skipped (label/parquet length mismatch)", flush=True)

    # --- meta ---
    # Clone info.json from kai0_base, fix counts
    info = json.loads((BASE_SRC / "meta" / "info.json").read_text())
    info["total_episodes"] = new_ep
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = new_ep * len(CAMERAS)
    info["total_chunks"] = 1
    info["chunks_size"] = max(1000, new_ep)
    info["splits"] = {"train": f"0:{new_ep}"}
    # 改动1: 不再 pop intervention(现予保留)。intervention/dagger_frame_class 作为 parquet 列携带,
    # 与源 dagger 一致(info.features 中未声明 → LeRobot 容忍未声明列, 训练 repack 只取 structure keys)。
    for k in ("observation.depth.top_head",):
        info.get("features", {}).pop(k, None)
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    (OUT / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n"
    )

    print(f"  → {OUT} ({new_ep} ep / {total_frames} frames)", flush=True)

    # --- norm_stats ---
    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)

    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()

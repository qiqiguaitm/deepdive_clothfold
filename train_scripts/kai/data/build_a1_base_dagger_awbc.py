#!/usr/bin/env python3
"""Build A1_base_dagger_awbc — Task_A1 (细长夹爪) AWBC warm-start 训练集.

plan: docs/training/future_plans/plans/pi05_task_a1_awbc_gripper_adapt_plan.md §3.1

合并 Task_A1 base(全部日期, 428 pq) + dagger(仅 2026-07-24, 42 ep) 成一个 LeRobot v2.1 集,
逐帧打 AWBC 标签(positive⟺人控), 裁 dagger 内部 class0 长静止段, 3 相机, 重编号, 重算 norm。

打标(positive ⟺ 人在控制, 与 init 模型 pi05_v4_awbc 的 AWBC prompt 格式一致):
  base(专家遥操, 全部)                 → task_index=1 positive
  dagger intervention==1(人控纠错, 含抓取) → task_index=1 positive
  dagger intervention==0(机器人自主)      → task_index=0 negative
  tasks.jsonl = {0:"...Advantage: negative", 1:"...Advantage: positive"}

裁剪(§2.5/§3.1, 仅 dagger, 不裁 base, 不裁 class1 人控):
  dagger 内 (dagger_frame_class==0 且 臂静止) 的 连续 run 长度 > STATIC_RUN_MIN(60帧=2s@30fps) → 丢弃整段。
  臂静止判据 = launchtrim 的 arm_moving_mask(ARM 关节 5帧平滑速度 <= THR); 只对 class0 生效,
  天然避开"臂速误杀抓取"(class1 grasp 不裁)。裁帧后重编码视频保持 parquet↔video 对齐。

val 留出(plan §4 T1: 看 warm-start×新norm 的早期 MAE 重对齐): 从 base 最后一天(07-24)留 VAL_N ep,
  单独建 A1_val(positive prompt), 不进 train, 不参与 norm。

相机: 仅 init 模型用的 3 路(top_head/hand_left/hand_right); 忽略 Task_A1 新增的 mid_head + depth。
norm: compute_norm_stats(train, action_dim=32)(细长夹爪值域变了必须重算, 不复用 init 的 norm)。

Run (North-E, 数据在那里):
  KAI0_ROOT=/vePFS-North-E/vis_robot/workspace/deepdive_kai0/kai0 \
  .venv/bin/python train_scripts/kai/data/build_a1_base_dagger_awbc.py [--dry-run] [--nproc 48]
"""
from __future__ import annotations

import argparse, json, os, shutil, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_no_release import per_episode_stats, _select_job  # noqa: E402
from launchpoint_trim_dagger import ARM, THR                 # noqa: E402

ROOT = Path(os.environ.get("KAI0_ROOT", "/vePFS/tim/workspace/deepdive_kai0/kai0"))
TASK_A1 = ROOT / "data" / "Task_A1"
BASE_ROOT = TASK_A1 / "base" / "v4"
DAGGER_ROOT = TASK_A1 / "dagger" / "v4"
DAGGER_DATE = "2026-07-24-v4"                 # 仅用 07-24(07-23 全自主已丢, 见 §2.5)
OUT = TASK_A1 / "self_built" / "A1_base_dagger_awbc"
OUT_VAL = TASK_A1 / "self_built" / "A1_val"

CAMERAS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
FPS = 30
CHUNK = 0
STATIC_RUN_MIN = 60          # >2s@30fps class0 静止 run 才裁
VAL_N = 5                    # base 07-24 留出 ep 数(val)
PROMPT_BASE = "Flatten and fold the cloth"
NEG = f"{PROMPT_BASE}. Advantage: negative"
POS = f"{PROMPT_BASE}. Advantage: positive"
KEEP_COLS = [
    "observation.state", "action", "timestamp", "frame_index",
    "episode_index", "index", "task_index",
    "intervention", "dagger_frame_class",     # dagger 有; base 无(下方填默认)
]


def arm_moving_mask(action: np.ndarray) -> np.ndarray:
    """launchtrim 逐帧运动判据: ARM 关节 5帧平滑速度 > THR(排除夹爪)。True=在动。"""
    v = np.linalg.norm(np.diff(action[:, ARM], axis=0), axis=1)
    v = np.concatenate([[0.0], v])
    vbar = np.convolve(v, np.ones(5) / 5, mode="same")
    return vbar > THR


def class0_static_keep(cls: np.ndarray, action: np.ndarray) -> np.ndarray:
    """返回 keep 布尔 mask: 丢弃 (class0 且 臂静止) 的 >STATIC_RUN_MIN 连续 run。"""
    n = len(cls)
    moving = arm_moving_mask(action)
    kill_cand = (cls == 0) & (~moving)
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n:
        if kill_cand[i]:
            j = i
            while j < n and kill_cand[j]:
                j += 1
            if (j - i) > STATIC_RUN_MIN:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return keep


def find_video(root: Path, chunk_dir: str, local_ep: int, cam: str) -> Path:
    for c in (cam, cam.replace("observation.images.", "")):
        p = root / "videos" / chunk_dir / c / f"episode_{local_ep:06d}.mp4"
        if p.exists() or p.is_symlink():
            return p
    raise FileNotFoundError(f"video 缺失: {root} {chunk_dir} {cam} ep{local_ep}")


def discover():
    """返回 [(group, src_root, chunk_dir, local_ep, pq_path)] 按 base(按日期)→dagger 顺序。
    并标出 val ep(base 07-24 的最后 VAL_N 个)。"""
    items = []
    for date_dir in sorted(BASE_ROOT.glob("*-v4")):
        pqs = sorted((date_dir / "data" / "chunk-000").glob("*.parquet"))
        for p in pqs:
            local_ep = int(p.stem.replace("episode_", ""))
            items.append(["base", date_dir, "chunk-000", local_ep, p, date_dir.name])
    dgd = DAGGER_ROOT / DAGGER_DATE
    for p in sorted((dgd / "data" / "chunk-001").glob("*.parquet")):
        local_ep = int(p.stem.replace("episode_", ""))
        items.append(["dagger", dgd, "chunk-001", local_ep, p, DAGGER_DATE])
    # val = base 07-24 的最后 VAL_N 个 ep
    base_0724 = [it for it in items if it[0] == "base" and it[5] == "2026-07-24-v4"]
    val_keys = {(it[1], it[3]) for it in base_0724[-VAL_N:]} if len(base_0724) >= VAL_N else set()
    return items, val_keys


def load_labeled(group, pq_path):
    """读源 parquet, 选列, 补 base 缺列, 裁 dagger 静止段。
    返回 (df, task_index_array, keep_idx, raw_n)。"""
    df = pd.read_parquet(pq_path)
    raw_n = len(df)
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
    if group == "base":
        # 专家遥操示范 → 全 positive; 补两列保证 schema 一致(不进 info.features)
        df["intervention"] = np.int64(1)
        df["dagger_frame_class"] = np.int64(5)      # CLASS_DEMO(base), 不参与裁剪/上采样
        keep_idx = np.arange(raw_n)
        ti = np.ones(raw_n, dtype=np.int64)         # positive
    else:
        df["intervention"] = df["intervention"].astype("int64")
        df["dagger_frame_class"] = df["dagger_frame_class"].astype("int64")
        cls = df["dagger_frame_class"].to_numpy()
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        keep = class0_static_keep(cls, act)
        keep_idx = np.where(keep)[0]
        df = df.iloc[keep_idx].reset_index(drop=True)
        ti = df["intervention"].to_numpy().astype(np.int64)   # positive⟺人控
    return df, ti, keep_idx, raw_n


def write_episode(df, ti, out_dir, new_ep, total_frames):
    n = len(df)
    df = df.copy()
    df["task_index"] = ti
    df["episode_index"] = np.int64(new_ep)
    df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
    df["frame_index"] = np.arange(n, dtype=np.int64)
    df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS)
    outp = out_dir / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), outp)
    return outp, n, df


def finalize_meta(out_dir, eps_meta, stats_out, new_ep, total_frames, tasks):
    info = json.loads((BASE_ROOT / "2026-07-21-v4" / "meta" / "info.json").read_text())
    info.update(total_episodes=new_ep, total_frames=total_frames, total_tasks=len(tasks),
                total_videos=new_ep * len(CAMERAS), total_chunks=1,
                chunks_size=max(1000, new_ep), splits={"train": f"0:{new_ep}"})
    # 只保留 3 相机: 去掉 Task_A1 新增的 mid_head + depth
    for k in ("observation.images.mid_head", "observation.depth.top_head"):
        info.get("features", {}).pop(k, None)
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (out_dir / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    with (out_dir / "meta" / "tasks.jsonl").open("w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--nproc", type=int, default=48)
    a = ap.parse_args()

    items, val_keys = discover()
    nb = sum(1 for x in items if x[0] == "base")
    nd = sum(1 for x in items if x[0] == "dagger")
    print(f"discovered base={nb} dagger={nd}  val_holdout={len(val_keys)} (base 07-24 last {VAL_N})",
          flush=True)

    if a.dry_run:
        # 统计裁剪量 + 正负比, 不写盘
        trimmed = kept_f = base_f = pos_f = neg_f = 0
        for grp, src_root, chunk_dir, local_ep, pqp, date_name in items:
            if (src_root, local_ep) in val_keys:
                continue
            df, ti, keep_idx, raw_n = load_labeled(grp, pqp)
            if grp == "dagger":
                trimmed += raw_n - len(df)
            else:
                base_f += len(df)
            kept_f += len(df)
            pos_f += int((ti == 1).sum()); neg_f += int((ti == 0).sum())
        print(f"  train frames(kept)={kept_f}  base={base_f}  dagger_trimmed_frames={trimmed}")
        print(f"  positive={pos_f} ({pos_f/max(1,kept_f):.1%})  negative={neg_f} ({neg_f/max(1,kept_f):.1%})")
        print("dry-run: nothing written")
        return

    for d in (OUT, OUT_VAL):
        if d.exists():
            shutil.rmtree(d)
        (d / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
        (d / "meta").mkdir()

    # ---- train + val 分别累积 ----
    tr_eps, tr_stats, tr_video_jobs = [], [], []
    va_eps, va_stats = [], []
    tr_ep = tr_frames = 0
    va_ep = va_frames = 0
    dropped = 0

    for grp, src_root, chunk_dir, local_ep, pqp, date_name in items:
        is_val = (src_root, local_ep) in val_keys
        df, ti, keep_idx, raw_n = load_labeled(grp, pqp)
        n = len(df)
        if n < 30:
            print(f"  drop {grp} {date_name} ep{local_ep}: 裁后仅 {n} 帧", flush=True)
            dropped += 1
            continue
        was_trimmed = (grp == "dagger" and len(keep_idx) != raw_n)

        out_dir = OUT_VAL if is_val else OUT
        new_ep = va_ep if is_val else tr_ep
        base_frames = va_frames if is_val else tr_frames
        outp, n, final_df = write_episode(df, ti, out_dir, new_ep, base_frames)

        # videos:
        #   train → 一律重编码(⚠️ Task_A1 原始录制 base 视频 keyframes=0 → LeRobot seek 解码失败 →
        #           dataloader 大量 skip; 重编码 libx264 加关键帧修复。base keep_idx=全帧, dagger=裁后帧)
        #   val   → symlink 即可(inline_eval 用顺序 decode, 0 关键帧也能全解码)
        for cam in CAMERAS:
            sv = find_video(src_root, chunk_dir, local_ep, cam)
            dv = out_dir / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            if is_val:
                os.symlink(str(sv.resolve()), dv)
            else:
                tr_video_jobs.append((str(sv.resolve()), str(dv), keep_idx.copy(), n))

        em = {"episode_index": new_ep, "tasks": [POS if grp == "base" else "mixed"],
              "length": n, "group": grp, "src_date": date_name, "src_ep": local_ep,
              "trimmed": was_trimmed}
        st = {"episode_index": new_ep, "stats": per_episode_stats(final_df)}
        if is_val:
            va_eps.append(em); va_stats.append(st); va_frames += n; va_ep += 1
        else:
            tr_eps.append(em); tr_stats.append(st); tr_frames += n; tr_ep += 1

    print(f"  train: {tr_ep} ep / {tr_frames} frames; val: {va_ep} ep / {va_frames} frames; "
          f"dropped={dropped}", flush=True)

    finalize_meta(OUT, tr_eps, tr_stats, tr_ep, tr_frames, [NEG, POS])
    finalize_meta(OUT_VAL, va_eps, va_stats, va_ep, va_frames, [NEG, POS])

    # ---- 裁剪的 dagger 视频重编码(并行) ----
    if tr_video_jobs:
        print(f"  re-encoding {len(tr_video_jobs)} train videos (base+dagger, 加关键帧修 seek; nproc={a.nproc})...", flush=True)
        with Pool(a.nproc) as pool_:
            for i, _ in enumerate(pool_.imap_unordered(_select_job, tr_video_jobs, chunksize=4), 1):
                if i % 30 == 0:
                    print(f"    video {i}/{len(tr_video_jobs)}", flush=True)
        print(f"  video re-encode done ({len(tr_video_jobs)})", flush=True)

    # ---- norm(仅 train, action_dim=32; 细长夹爪重算)----
    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats on TRAIN (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)

    print(f"  → {OUT} ({tr_ep} ep / {tr_frames} frames)  + val {OUT_VAL} ({va_ep} ep)", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()

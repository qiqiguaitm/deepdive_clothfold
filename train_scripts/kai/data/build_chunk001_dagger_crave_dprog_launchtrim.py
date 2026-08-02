#!/usr/bin/env python3
"""Build A_v4_chunk001_dagger_crave_dprog_launchtrim: fixes the freeze-inducing label of
A_v4_chunk001_dagger_crave_labeled (see dagger_launchpoint_trim_freeze_fix_plan §9).

Two fixes vs the old build (build_chunk001_dagger_crave_labeled.py):
  1. advantage = forward Δprogress (spg[t+H]-spg[t]) instead of progress LEVEL.
     Old: absolute_advantage ≡ stage_progress_gt (corr 1.0) → top-30% = last 30% of the task =
     high-progress STATIC settle frames → freeze. New: Δprogress rewards frames that ACTIVELY
     advance the task; static plateaus (settle/hesitation) get Δ≈0 → negative.
  2. launchtrim boundary trim on dagger (reuse launch_window: front hesitation + tail settle).
     base untrimmed (same single-variable policy as launchtrim).
  + velocity gate: a frame can only be POSITIVE if the arm is actually moving
     (launchtrim's smoothed vbar>THR, arm joints excl. gripper). Guarantees 0% static-positive —
     the property that predicts no-freeze. (Measured: old dagger 55.9% static-positive → 0.0%.)

task_index/tasks.jsonl written here directly (velocity gate not supported by discretize CLI).
Global top-(100-THRESH)% Δprogress threshold pooled over base+dagger, then gated.

Run (where labels + chunk-001 source live; videos re-encoded for trimmed dagger):
  KAI0_ROOT=/vePFS/tim/workspace/deepdive_kai0/kai0 \
  .venv/bin/python train_scripts/kai/data/build_chunk001_dagger_crave_dprog_launchtrim.py [--dry-run]
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
from launchpoint_trim_dagger import launch_window, ARM, THR   # noqa: E402

ROOT = Path(os.environ.get("KAI0_ROOT", "/vePFS/tim/workspace/deepdive_kai0/kai0"))
WORKSPACE = ROOT.parent
TASK_A = ROOT / "data" / "Task_A"
BASE_SRC = TASK_A / "kai0_base"
DAGGER_V4 = TASK_A / "vis_dagger" / "v4"
LABEL_DIR = WORKSPACE / "lmvla" / "crave" / "temp" / "crave_ae_labels" / "chunk001_val"
OUT = TASK_A / "self_built" / "A_v4_chunk001_dagger_crave_dprog_launchtrim"

CAMERAS = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)
FPS = 30
CHUNK = 0
PROMPT = "Flatten and fold the cloth."
H = 50          # Δprogress horizon (= action-chunk length; frames looked ahead)
THRESH = 30     # top-30% of Δprogress → positive (before velocity gate)
KEEP_COLS = [
    "observation.state", "action", "timestamp", "frame_index",
    "episode_index", "index", "task_index",
]


def decode_dagger_global_id(global_id: int):
    dh = global_id // 10000
    local_ep = global_id % 10000
    return f"2026-{dh // 100:02d}-{dh % 100:02d}-v4", local_ep


def _find_pq(data_dir: Path, ep_id: int) -> Path | None:
    p = data_dir / f"chunk-{ep_id // 1000:03d}" / f"episode_{ep_id:06d}.parquet"
    return p if p.exists() else None


def discover_episodes():
    items = []
    bl = LABEL_DIR / "base"
    if bl.is_dir():
        for npy in sorted(bl.glob("ep*.npy")):
            ep_id = int(npy.stem.replace("ep", ""))
            pqp = _find_pq(BASE_SRC / "data", ep_id)
            if pqp:
                items.append(("base", ep_id, ep_id, npy, pqp, None))
            else:
                print(f"  skip base ep{ep_id}: parquet 缺失", flush=True)
    dl = LABEL_DIR / "dagger"
    if dl.is_dir():
        for npy in sorted(dl.glob("ep*.npy")):
            gid = int(npy.stem.replace("ep", ""))
            date_dir, local_ep = decode_dagger_global_id(gid)
            pqp = DAGGER_V4 / date_dir / "data" / "chunk-001" / f"episode_{local_ep:06d}.parquet"
            if pqp.exists():
                items.append(("dagger", gid, local_ep, npy, pqp, date_dir))
            else:
                print(f"  skip dagger {date_dir} ep{local_ep}: {pqp} 缺失", flush=True)
    nb = sum(1 for x in items if x[0] == "base")
    nd = sum(1 for x in items if x[0] == "dagger")
    print(f"  discovered base={nb} dagger={nd}", flush=True)
    return items


def find_video(group, src_date_dir, ep_id, cam) -> Path:
    if group == "base":
        root, chunk_dir = BASE_SRC, f"chunk-{ep_id // 1000:03d}"
    else:
        root, chunk_dir = DAGGER_V4 / src_date_dir, "chunk-001"
    for c in (cam, cam.replace("observation.images.", "")):
        p = root / "videos" / chunk_dir / c / f"episode_{ep_id:06d}.mp4"
        if p.exists() or p.is_symlink():
            return p
    raise FileNotFoundError(f"video 缺失: {group} {src_date_dir or 'base'} {cam} ep{ep_id}")


def arm_moving_mask(action: np.ndarray) -> np.ndarray:
    """launchtrim's per-frame motion test: 5-frame smoothed arm speed > THR (gripper excluded)."""
    v = np.linalg.norm(np.diff(action[:, ARM], axis=0), axis=1)
    v = np.concatenate([[0.0], v])
    vbar = np.convolve(v, np.ones(5) / 5, mode="same")
    return vbar > THR


def dprogress(spg: np.ndarray, h: int) -> np.ndarray:
    T = len(spg)
    idx = np.minimum(np.arange(T) + h, T - 1)
    return (spg[idx] - spg).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-norm", action="store_true")
    ap.add_argument("--nproc", type=int, default=48)
    a = ap.parse_args()

    items = discover_episodes()
    if not items:
        print("FATAL: 无 episode", file=sys.stderr); sys.exit(1)

    if a.dry_run:
        # simulate trim + Δprogress + gate, report static-positive (freeze predictor)
        all_adv, gate_all, grp_all, kept, dropped = [], [], [], 0, 0
        for grp, ep_id, local_ep, npy, pqp, src_date in items:
            spg = np.load(npy).astype(np.float32)
            df = pd.read_parquet(pqp)
            if len(spg) != len(df):
                continue
            act = np.stack(df["action"].to_numpy()).astype(np.float64)
            if grp == "dagger":
                w = launch_window(act)
                if w is None:
                    dropped += 1; continue
                s, e = w; spg = spg[s:e]; act = act[s:e]
            kept += 1
            adv = dprogress(spg, H)
            gate = arm_moving_mask(act)
            all_adv.append(adv); gate_all.append(gate); grp_all.append(grp)
        pool = np.concatenate(all_adv); thr = np.percentile(pool, 100 - THRESH)
        # per-group static-positive
        for want in ("dagger", "base"):
            sp = pos = tot = 0
            for adv, gate, g in zip(all_adv, gate_all, grp_all):
                if g != want:
                    continue
                p = (adv >= thr) & gate
                static = ~gate
                sp += int((static & p).sum()); pos += int(p.sum()); tot += len(adv)
            print(f"  {want}: kept posfrac={pos/max(1,tot):.3f} static&pos/pos={100*sp/max(1,pos):.1f}%")
        print(f"  Δprog top-{THRESH}% threshold={thr:.4f}; dagger dropped(pure-hold/short)={dropped}")
        print("dry-run: nothing written")
        return

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / f"chunk-{CHUNK:03d}").mkdir(parents=True)
    (OUT / "meta").mkdir()

    # ---- pass 1: trim + write parquet (placeholder task_index) + queue video jobs ----
    eps_meta, adv_arrays, gate_arrays, parquet_paths = [], [], [], []
    video_jobs = []
    total_frames = new_ep = dropped = skipped_mismatch = 0

    for grp, ep_id, local_ep, npy, pqp, src_date in items:
        spg = np.load(npy).astype(np.float32)
        df = pd.read_parquet(pqp)
        df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
        if len(spg) != len(df):
            print(f"  WARN {grp} ep{ep_id}: label {len(spg)} != pq {len(df)} — skip", flush=True)
            skipped_mismatch += 1; continue

        act_full = np.stack(df["action"].to_numpy()).astype(np.float64)
        s, e = 0, len(df)
        if grp == "dagger":
            w = launch_window(act_full)
            if w is None:
                dropped += 1; continue
            s, e = w
            df = df.iloc[s:e].reset_index(drop=True)
            spg = spg[s:e]
            act_full = act_full[s:e]

        n = len(df)
        adv = dprogress(spg, H)
        gate = arm_moving_mask(act_full)

        df["stage_progress_gt"] = spg
        df["absolute_advantage"] = adv            # = Δprogress (audit column)
        df["episode_index"] = np.int64(new_ep)
        df["index"] = np.arange(total_frames, total_frames + n, dtype=np.int64)
        df["frame_index"] = np.arange(n, dtype=np.int64)
        df["timestamp"] = (np.arange(n, dtype=np.float32) / FPS)
        df["task_index"] = np.int64(0)            # placeholder, filled pass 2

        outp = OUT / "data" / f"chunk-{CHUNK:03d}" / f"episode_{new_ep:06d}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), outp)

        for cam in CAMERAS:
            sv = find_video(grp, src_date, local_ep, cam)
            dv = OUT / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{new_ep:06d}.mp4"
            dv.parent.mkdir(parents=True, exist_ok=True)
            if grp == "dagger":
                video_jobs.append((str(sv.resolve()), str(dv), np.arange(s, e), n))
            else:
                os.symlink(str(sv.resolve()), dv)   # base untrimmed → symlink

        eps_meta.append({"episode_index": new_ep, "tasks": [PROMPT], "length": n,
                         "src_ep": ep_id, "group": grp, "src_date": src_date,
                         "trim": [int(s), int(e)] if grp == "dagger" else None})
        adv_arrays.append(adv); gate_arrays.append(gate); parquet_paths.append(outp)
        total_frames += n; new_ep += 1

    print(f"  pass1: {new_ep} ep / {total_frames} frames (dagger dropped={dropped}, "
          f"mismatch={skipped_mismatch})", flush=True)

    # ---- pass 2: global Δprogress top-30% threshold + velocity gate → task_index ----
    pool = np.concatenate(adv_arrays)
    thr = float(np.percentile(pool, 100 - THRESH))
    print(f"  Δprogress top-{THRESH}% threshold = {thr:.5f}", flush=True)
    stats_out = []
    pos_tot = static_pos = 0
    for outp, adv, gate, em in zip(parquet_paths, adv_arrays, gate_arrays, eps_meta):
        ti = ((adv >= thr) & gate).astype(np.int64)
        df = pd.read_parquet(outp)
        df["task_index"] = ti
        df.to_parquet(outp, index=False)
        stats_out.append({"episode_index": em["episode_index"], "stats": per_episode_stats(df)})
        pos_tot += int(ti.sum()); static_pos += int((ti.astype(bool) & (~gate)).sum())
    print(f"  positive frames={pos_tot} ({100*pos_tot/total_frames:.1f}%), "
          f"static&positive={static_pos} (must be 0)", flush=True)

    # ---- meta ----
    info = json.loads((BASE_SRC / "meta" / "info.json").read_text())
    info.update(total_episodes=new_ep, total_frames=total_frames, total_tasks=2,
                total_videos=new_ep * len(CAMERAS), total_chunks=1,
                chunks_size=max(1000, new_ep), splits={"train": f"0:{new_ep}"})
    for k in ("observation.depth.top_head", "intervention"):
        info.get("features", {}).pop(k, None)
    (OUT / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (OUT / "meta" / "episodes.jsonl").open("w") as f:
        for em in eps_meta:
            f.write(json.dumps(em) + "\n")
    with (OUT / "meta" / "episodes_stats.jsonl").open("w") as f:
        for st in stats_out:
            f.write(json.dumps(st) + "\n")
    base = PROMPT.rstrip(".,")
    with (OUT / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": f"{base}. Advantage: negative"}) + "\n")
        f.write(json.dumps({"task_index": 1, "task": f"{base}. Advantage: positive"}) + "\n")

    # ---- videos: re-encode trimmed dagger clips (parallel; base already symlinked) ----
    if video_jobs:
        print(f"  re-encoding {len(video_jobs)} trimmed dagger videos (nproc={a.nproc})...", flush=True)
        with Pool(a.nproc) as pool_:
            for i, _ in enumerate(pool_.imap_unordered(_select_job, video_jobs, chunksize=4), 1):
                if i % 300 == 0:
                    print(f"    video {i}/{len(video_jobs)}", flush=True)
        print(f"  video re-encode done ({len(video_jobs)})", flush=True)

    if not a.no_norm:
        from norm_stats_from_dataset import compute_norm_stats
        print("  computing norm_stats (action_dim=32)...", flush=True)
        compute_norm_stats(str(OUT), action_dim=32)

    print(f"  → {OUT} ({new_ep} ep / {total_frames} frames)", flush=True)
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()

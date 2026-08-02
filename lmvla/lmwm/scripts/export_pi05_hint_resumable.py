#!/usr/bin/env python
"""Resume-safe episode-sharded pi05 hint export."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_pi05_hint import _load_grid_npz, grid_to_hint, load_lmwm


def episode_number(path: str) -> int:
    return int(os.path.basename(path)[2:-4])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    gen, prd, din, code_dim, mdn_k = load_lmwm(args.ckpt, args.device)
    print(
        f"[lmwm] din={din} code_dim={code_dim} MDN-K={mdn_k} "
        f"export-K={args.K} device={args.device}",
        flush=True,
    )

    shard_root = os.path.join(args.out, "_episode_shards")
    suites = sorted(
        name
        for name in os.listdir(args.feat_root)
        if os.path.isdir(os.path.join(args.feat_root, name))
    )
    bad: list[tuple[str, str]] = []
    total = skipped = completed = 0
    for suite in suites:
        episode_paths = sorted(
            glob.glob(os.path.join(args.feat_root, suite, "ep*.npz")),
            key=episode_number,
        )
        suite_out = os.path.join(shard_root, suite)
        os.makedirs(suite_out, exist_ok=True)
        for episode_path in episode_paths:
            total += 1
            ep = episode_number(episode_path)
            shard = os.path.join(suite_out, f"ep{ep}.npz")
            if os.path.isfile(shard):
                skipped += 1
                continue
            try:
                grid, frame_index = _load_grid_npz(episode_path, din)
                hints = []
                for start in range(0, len(grid), args.batch):
                    batch = torch.from_numpy(grid[start : start + args.batch]).to(args.device)
                    hints.append(grid_to_hint(gen, prd, batch, args.K).cpu().numpy())
                hint = np.concatenate(hints).astype(np.float16)
                tmp = shard + ".tmp.npz"
                np.savez_compressed(tmp, frame_index=frame_index, hint=hint)
                os.replace(tmp, shard)
                completed += 1
                if completed % 50 == 0:
                    print(
                        f"[export] completed={completed} skipped={skipped} "
                        f"seen={total} suite={suite} ep={ep}",
                        flush=True,
                    )
            except Exception as exc:
                bad.append((episode_path, f"{type(exc).__name__}: {exc}"))
                print(f"[bad] {episode_path}: {bad[-1][1]}", flush=True)

    bad_path = os.path.join(args.out, "bad_episode_paths.txt")
    os.makedirs(args.out, exist_ok=True)
    with open(bad_path, "w", encoding="utf-8") as handle:
        for path, error in bad:
            handle.write(f"{path}\t{error}\n")
    if bad:
        raise RuntimeError(f"{len(bad)} corrupt episodes; see {bad_path}")

    all_suite: list[str] = []
    all_ep: list[int] = []
    all_frame_index: list[np.ndarray] = []
    all_hint: list[np.ndarray] = []
    for suite in suites:
        for shard in sorted(glob.glob(os.path.join(shard_root, suite, "ep*.npz")), key=episode_number):
            ep = episode_number(shard)
            with np.load(shard) as data:
                frame_index = data["frame_index"].astype(np.int64)
                hint = data["hint"].astype(np.float16)
            all_suite.extend([suite] * len(hint))
            all_ep.extend([ep] * len(hint))
            all_frame_index.append(frame_index)
            all_hint.append(hint)

    hint = np.concatenate(all_hint)
    output = os.path.join(args.out, "hint.npz")
    tmp_output = output + ".tmp.npz"
    np.savez_compressed(
        tmp_output,
        suite=np.asarray(all_suite),
        episode_index=np.asarray(all_ep, dtype=np.int64),
        frame_index=np.concatenate(all_frame_index).astype(np.int64),
        hint=hint,
    )
    os.replace(tmp_output, output)
    metadata = {
        "ckpt": args.ckpt,
        "feat_root": args.feat_root,
        "din": din,
        "code_dim": code_dim,
        "K": args.K,
        "n_frames": int(len(hint)),
        "hint_shape": list(hint.shape),
        "episode_shards": total,
        "note": "resume-safe episode-sharded export; hint = pooled predicted next grid",
    }
    with open(os.path.join(args.out, "_env.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"[done] {total} episodes {len(hint)} frames -> {output}", flush=True)


if __name__ == "__main__":
    main()

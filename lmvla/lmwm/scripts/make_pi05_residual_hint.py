#!/usr/bin/env python
"""Convert absolute pi05 LMWM hints to residual hints.

For every (suite, episode_index, frame_index), residual = hint - mean(current_grid).
The input feature root must use <root>/<suite>/ep{E}.npz with key "grid".
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hint", required=True)
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    z = np.load(args.hint, allow_pickle=True)
    suites = z["suite"].astype(str)
    eps = z["episode_index"].astype(np.int64)
    fis = z["frame_index"].astype(np.int64)
    hints = z["hint"].astype(np.float32)
    out_hint = np.empty_like(hints, dtype=np.float16)

    cache: dict[tuple[str, int], np.ndarray] = {}
    for i, (suite, ep, fi) in enumerate(zip(suites, eps, fis, strict=True)):
        key = (suite, int(ep))
        grid = cache.get(key)
        if grid is None:
            path = os.path.join(args.feat_root, suite, f"ep{int(ep)}.npz")
            grid = np.load(path)["grid"].astype(np.float32)
            grid = grid.mean(axis=1)
            cache[key] = grid
        out_hint[i] = (hints[i] - grid[int(fi)]).astype(np.float16)
        if len(cache) > 32:
            cache.pop(next(iter(cache)))

    os.makedirs(args.out, exist_ok=True)
    out_npz = os.path.join(args.out, "hint.npz")
    np.savez_compressed(
        out_npz,
        suite=suites,
        episode_index=eps,
        frame_index=fis,
        hint=out_hint,
    )
    env = {
        "source_hint": args.hint,
        "feat_root": args.feat_root,
        "n_frames": int(len(out_hint)),
        "hint_shape": list(out_hint.shape),
        "note": "residual = absolute_hint - mean(current_grid_tokens)",
    }
    with open(os.path.join(args.out, "_env.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print(f"[done] residual hint {out_hint.shape} -> {out_npz}", flush=True)


if __name__ == "__main__":
    main()

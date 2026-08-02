#!/usr/bin/env python3
"""Standalone fast norm stats for official-aligned RoboTwin pi05.

No openpi import: importing the full training config stack is slow on the Volc
image. The output JSON matches openpi.shared.normalize.save:
{"norm_stats": {"state": {"mean": ..., "std": ..., "q01": ..., "q99": ...}, ...}}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class RunningStats:
    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        if x.size == 0:
            return
        batch_n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.n + batch_n
        self.mean = self.mean + delta * batch_n / total
        self.m2 = self.m2 + batch_m2 + delta**2 * self.n * batch_n / total
        self.n = total

    def as_json(self) -> dict[str, list[float]]:
        var = self.m2 / max(self.n - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-12))
        # q01/q99 are not used by these configs (use_quantile_norm=False), but
        # keeping plausible values preserves schema compatibility.
        q01 = self.mean - 2.326347874 * std
        q99 = self.mean + 2.326347874 * std
        return {
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }


def pad_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    if x.shape[-1] >= dim:
        return x[..., :dim].astype(np.float32, copy=False)
    pad = np.zeros((*x.shape[:-1], dim - x.shape[-1]), dtype=np.float32)
    return np.concatenate([x.astype(np.float32, copy=False), pad], axis=-1)


def stack_column(df: pd.DataFrame, name: str) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in df[name].to_numpy()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument(
        "--absolute-actions",
        action="store_true",
        help="Accumulate raw absolute action chunks instead of Aloha joint deltas.",
    )
    args = ap.parse_args()

    repo = Path(args.repo)
    out = Path(args.out)
    files = sorted((repo / "data").rglob("*.parquet"))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files under {repo / 'data'}")

    # Aloha mask: delta for 6 arm joints per side, absolute grippers at dims 6/13.
    mask14 = np.array([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
    delta_mask = np.zeros(args.action_dim, dtype=bool)
    delta_mask[:14] = mask14

    state_stats = RunningStats(args.action_dim)
    action_stats = RunningStats(args.action_dim)

    print(
        f"repo={repo} files={len(files)} action_dim={args.action_dim} "
        f"horizon={args.horizon} out={out}",
        flush=True,
    )
    for pq in tqdm(files, desc="robotwin norm files"):
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        state = pad_to_dim(stack_column(df, "observation.state"), args.action_dim)
        action = pad_to_dim(stack_column(df, "action"), args.action_dim)
        if len(state) == 0:
            continue

        state_stats.update(state)
        if args.absolute_actions:
            # Match LeRobot's dataset-level MEAN_STD processor: each recorded
            # action frame contributes exactly once, independent of chunking.
            action_stats.update(action)
            continue

        tail = np.repeat(action[-1:], args.horizon, axis=0)
        padded = np.concatenate([action, tail], axis=0)
        starts = np.arange(len(action), dtype=np.int64)[:, None]
        offsets = np.arange(args.horizon, dtype=np.int64)[None, :]
        chunks = padded[starts + offsets].astype(np.float32, copy=True)
        chunks[..., delta_mask] -= state[:, None, delta_mask]
        action_stats.update(chunks)

    payload = {"norm_stats": {"state": state_stats.as_json(), "actions": action_stats.as_json()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_stats.json").write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {out / 'norm_stats.json'} state_n={state_stats.n} action_n={action_stats.n}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-loud validation for converted XVLA EE6D LeRobot datasets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def fail(message: str) -> None:
    raise SystemExit(f"EE6D PREFLIGHT FAILED: {message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--open-m", type=float)
    ap.add_argument("--close-m", type=float)
    args = ap.parse_args()

    root = args.root
    info = json.loads((root / "meta" / "info.json").read_text())
    for key in ("observation.state", "action"):
        shape = info.get("features", {}).get(key, {}).get("shape")
        if shape != [20]:
            fail(f"{key} metadata shape={shape}, expected [20]")

    expected_names = [
        f"{side}_{x}" for side in ("left", "right")
        for x in ("x", "y", "z", "r00", "r01", "r10", "r11", "r20", "r21", "grip")
    ]
    for key in ("observation.state", "action"):
        names = info["features"][key].get("names")
        if names and names != expected_names:
            fail(f"{key} names do not match interleaved Rot6D layout: {names}")

    if args.continuous:
        bounds = info.get("gripper_alpha_bounds")
        if not bounds:
            fail("continuous dataset has no gripper_alpha_bounds in info.json")
        if args.open_m is not None and not np.isclose(bounds.get("open_m"), args.open_m):
            fail(f"open bound {bounds.get('open_m')} != expected {args.open_m}")
        if args.close_m is not None and not np.isclose(bounds.get("close_m"), args.close_m):
            fail(f"close bound {bounds.get('close_m')} != expected {args.close_m}")

    tasks = {}
    with (root / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            tasks[int(item["task_index"])] = item["task"]
    if set(tasks) != {0, 1} or "negative" not in tasks[0].lower() or "positive" not in tasks[1].lower():
        fail(f"AWBC tasks mapping is not negative=0/positive=1: {tasks}")

    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        fail("no parquet files")

    rows = 0
    task_counts: Counter[int] = Counter()
    pos_min = np.full(6, np.inf)
    pos_max = np.full(6, -np.inf)
    grip_min = np.full(4, np.inf)
    grip_max = np.full(4, -np.inf)
    grip_sum = np.zeros(4)
    grip_sq_sum = np.zeros(4)
    grip_interior = np.zeros(4, dtype=np.int64)
    grip_values = [set() for _ in range(4)]
    max_rot_norm_err = 0.0
    max_rot_dot = 0.0
    grip_delta_sum = np.zeros(2)

    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action", "task_index"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 20 or action.shape != state.shape:
            fail(f"bad tensor shape in {path}: state={state.shape}, action={action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            fail(f"NaN/Inf in {path}")

        n = len(state)
        rows += n
        task_counts.update(map(int, table["task_index"].to_pylist()))
        xyz = np.concatenate((state[:, 0:3], state[:, 10:13]), axis=1)
        pos_min = np.minimum(pos_min, xyz.min(axis=0))
        pos_max = np.maximum(pos_max, xyz.max(axis=0))

        for array in (state, action):
            for start in (0, 10):
                rot = array[:, start + 3:start + 9].reshape(-1, 3, 2)
                norms = np.linalg.norm(rot, axis=1)
                dots = np.sum(rot[:, :, 0] * rot[:, :, 1], axis=1)
                max_rot_norm_err = max(max_rot_norm_err, float(np.abs(norms - 1).max()))
                max_rot_dot = max(max_rot_dot, float(np.abs(dots).max()))

        grips = np.stack((state[:, 9], state[:, 19], action[:, 9], action[:, 19]), axis=1)
        grip_min = np.minimum(grip_min, grips.min(axis=0))
        grip_max = np.maximum(grip_max, grips.max(axis=0))
        grip_sum += grips.sum(axis=0)
        grip_sq_sum += np.square(grips, dtype=np.float64).sum(axis=0)
        grip_interior += ((grips > 1e-4) & (grips < 1 - 1e-4)).sum(axis=0)
        for i in range(4):
            grip_values[i].update(np.round(grips[:, i], 4).tolist())
        grip_delta_sum += np.abs(action[:, [9, 19]] - state[:, [9, 19]]).sum(axis=0)

    if (np.abs(pos_min) > 2).any() or (np.abs(pos_max) > 2).any():
        fail(f"implausible FK xyz range: min={pos_min}, max={pos_max}")
    if max_rot_norm_err > 2e-4 or max_rot_dot > 2e-4:
        fail(f"invalid Rot6D: norm_err={max_rot_norm_err:.3g}, dot={max_rot_dot:.3g}")
    if set(task_counts) != {0, 1}:
        fail(f"dataset does not contain both AWBC classes: {dict(task_counts)}")

    if args.continuous:
        if (grip_min < -1e-6).any() or (grip_max > 1 + 1e-6).any():
            fail(f"continuous alpha outside [0,1]: min={grip_min}, max={grip_max}")
        if (grip_interior == 0).any() or any(len(v) < 20 for v in grip_values):
            fail(f"gripper was accidentally binarized: interior={grip_interior}, unique4={[len(v) for v in grip_values]}")

    grip_mean = grip_sum / rows
    grip_std = np.sqrt(np.maximum(grip_sq_sum / rows - np.square(grip_mean), 0))
    grip_delta = grip_delta_sum / rows
    if (grip_delta < 1e-4).any():
        fail(f"action gripper nearly copies state: mean_abs_delta={grip_delta}")

    print(f"EE6D PREFLIGHT PASS: {len(files)} parquets, {rows} rows")
    print(f"  xyz state min={pos_min.round(4).tolist()} max={pos_max.round(4).tolist()}")
    print(f"  Rot6D max_norm_err={max_rot_norm_err:.2e} max_dot={max_rot_dot:.2e}")
    print(f"  grip [stateL,stateR,actionL,actionR] min={grip_min.round(5).tolist()} "
          f"max={grip_max.round(5).tolist()} mean={grip_mean.round(5).tolist()} "
          f"std={grip_std.round(5).tolist()}")
    print(f"  grip interior={grip_interior.tolist()} unique@1e-4={[len(v) for v in grip_values]} "
          f"mean|action-state|={grip_delta.round(6).tolist()}")
    print(f"  AWBC task counts={dict(sorted(task_counts.items()))} prompts={tasks}")


if __name__ == "__main__":
    main()

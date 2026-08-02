#!/usr/bin/env python3
"""A1 (Conditioning arm) 打标: positive ⟺ 人在控制。

三范式对比实验 (awbc_three_paradigm_comparison_plan.md) 的 A1 臂。
取代 discretize(top-30% 进度)——那套是 B0 基线。本臂用**人控 vs 机器人**平衡二值:

  task_index = 1 (positive)  当 intervention == 1  → base(人遥操示范) + dagger 人控纠错(class1, 含抓取)
  task_index = 0 (negative)  当 intervention == 0  → dagger 机器人自主(class0 正常 + class2 临失败)

抓取(臂静止+夹爪合拢)落在 class1 人控 → 无论臂速一律 positive → 不会被速度门控误杀
(见 project_velocity_gate_kills_grasp / dagger_launchpoint_trim_freeze_fix_plan)。

推理永远喂 positive。用法:
  kai0/.venv/bin/python train_scripts/kai/data/relabel_human_awbc.py \
      --data kai0/data/Task_A/self_built/A_v4_chunk001_dagger_crave_human
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROMPT = "Flatten and fold the cloth."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(a.data)
    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "episode_*.parquet")))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")

    pos = tot = 0
    for f in files:
        df = pq.read_table(f).to_pandas()
        if "intervention" not in df.columns:
            raise SystemExit(f"FATAL: {f} 无 intervention 列 (build 未带改动1?)")
        ti = (np.asarray(df["intervention"].tolist()).astype(np.int64).ravel() == 1).astype(np.int64)
        pos += int(ti.sum()); tot += len(ti)
        if not a.dry_run:
            df["task_index"] = ti
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), f)

    tasks = root / "meta" / "tasks.jsonl"
    if not a.dry_run:
        tasks.write_text(
            json.dumps({"task_index": 0, "task": f"{PROMPT} Advantage: negative"}) + "\n"
            + json.dumps({"task_index": 1, "task": f"{PROMPT} Advantage: positive"}) + "\n"
        )
    print(f"{'[dry-run] ' if a.dry_run else ''}{len(files)} eps, positive(人控)={100*pos/tot:.1f}% ({pos}/{tot})")
    if not a.dry_run:
        print(f"  → task_index 重写完成; tasks.jsonl(pos/neg) → {tasks}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dagger_classify.py — 为 dagger 数据集逐帧打 dagger_frame_class 标签。

基于研究结论 (Sirius IJRR 2024, CR-DAgger arXiv 2506.16685):
  - preintv (接管前 ~15 帧) 标记但不训练
  - hesitation (接管后低速起手 ~0.5s) 裁掉/排除
  - stationary_tail (遥操结束后静止尾巴) 裁掉/排除
  - intv_core (果断遥操核心) 保留 + 上采样

帧类别:
  0 = robot         策略自主执行 (正常速度)
  1 = intv_core     人类遥操果断动作 (核心纠错信号)
  2 = preintv       接管前 N 帧 (机器人失败态, 不模仿)
  3 = hesitation    接管后低速迟疑起手 (遥操伪影, 排除)
  4 = stationary_tail  遥操结束后静止尾巴 (idle 伪影, 排除)
  5 = demo          base 示范 episode (全部帧)

当前数据格式:
  - dagger/ episodes: 全部帧 intervention=1 (纯人类遥操)
  - inference/ episodes: 全部帧 intervention=0 (纯策略执行)
  - base/ episodes: 无 intervention 列或全部 -1 (示范)
  - 暂不存在含 0→1 切换的单 episode (未来组合式记录会用到过渡检测逻辑)

用法:
  # 预览 (不改文件)
  python classify_dagger_frames.py /path/to/Task_A/dagger/v4/2026-07-13-v4/ --dry-run

  # 执行 (原地修改 parquet, 加 dagger_frame_class 列; 默认备份原文件)
  python classify_dagger_frames.py /path/to/Task_A/dagger/v4/2026-07-13-v4/

  # 处理 base 数据 (所有帧 → class 5)
  python classify_dagger_frames.py /path/to/Task_A/base/v4/2026-07-13-v4/ --subset base

  # 处理 inference 数据 (所有帧 → class 0, 但检测 tail stationary)
  python classify_dagger_frames.py /path/to/Task_A/inference/v4/2026-07-13-v4/ --subset inference

  # 不备份 (危险, 仅当已确认无误)
  python classify_dagger_frames.py ... --no-backup
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── 与 dataset_writer.py / build_no_release.py 共用常量 ──
ARM_DIMS = list(range(0, 6)) + list(range(7, 13))   # 12 arm dims (排除 gripper dims 6,13)
GRIP_DIMS = [6, 13]                                   # L/R gripper action dims
FPS = 30

# 迟疑检测: velocity 连续超过 HESITATION_THR 达 HESITATION_WIN 帧 = 迟疑结束
HESITATION_THR = 5e-3       # rad/frame — 低于此视为迟疑/慢速起手
HESITATION_WIN = 3          # 连续帧数
HESITATION_MAX = 30         # 最多标记这么多帧 (~1s @30Hz)

# 干预前窗口: 接管前这么多帧标记为 preintv
PREINTV_MARGIN = 15         # 帧 (~0.5s @30Hz)

# 静止尾巴: arm velocity 低于此 + gripper 静帧
STATIONARY_THR = 3e-3       # rad/frame — 与 TRIM_THR 一致

# 帧类别编码
CLASS_ROBOT = 0
CLASS_INTV_CORE = 1
CLASS_PREINTV = 2
CLASS_HESITATION = 3
CLASS_STATIONARY_TAIL = 4
CLASS_DEMO = 5

CLASS_NAMES = {
    0: "robot",
    1: "intv_core",
    2: "preintv",
    3: "hesitation",
    4: "stationary_tail",
    5: "demo",
}


def compute_arm_velocity(actions: np.ndarray) -> np.ndarray:
    """计算每帧的 12 维 arm velocity: mean(|Δaction[ARM_DIMS]|).

    Args:
        actions: shape (N, 14) — action 向量

    Returns:
        shape (N,) — 每帧的 arm velocity. 第 0 帧 velocity = 0.
    """
    if len(actions) < 2:
        return np.zeros(len(actions), dtype=np.float64)
    delta = np.abs(np.diff(actions[:, ARM_DIMS], axis=0))
    vel = np.zeros(len(actions), dtype=np.float64)
    vel[1:] = delta.mean(axis=1)
    return vel


def compute_gripper_velocity(actions: np.ndarray) -> np.ndarray:
    """计算每帧的 gripper velocity: max(|Δaction[GRIP_DIMS]|).

    Args:
        actions: shape (N, 14)

    Returns:
        shape (N,) — 每帧的 gripper velocity. 第 0 帧 velocity = 0.
    """
    if len(actions) < 2:
        return np.zeros(len(actions), dtype=np.float64)
    delta = np.abs(np.diff(actions[:, GRIP_DIMS], axis=0))
    vel = np.zeros(len(actions), dtype=np.float64)
    vel[1:] = delta.max(axis=1)
    return vel


def find_intervention_transitions(intervention: np.ndarray) -> list[dict]:
    """找到 intervention 列中的所有 0→1 和 1→0 切换点.

    Args:
        intervention: shape (N,) int array, -1=N/A, 0=policy, 1=human

    Returns:
        list of dicts: [{"frame": idx, "type": "takeover"|"release"}, ...]
    """
    transitions = []
    for i in range(1, len(intervention)):
        prev, curr = int(intervention[i - 1]), int(intervention[i])
        if prev == 0 and curr == 1:
            transitions.append({"frame": i, "type": "takeover"})  # 人类接管
        elif prev == 1 and curr == 0:
            transitions.append({"frame": i, "type": "release"})   # 交还策略
    return transitions


def classify_frames(
    actions: np.ndarray,
    intervention: np.ndarray | None = None,
    subset: str = "dagger",
) -> np.ndarray:
    """为 episode 的每一帧分配 dagger_frame_class.

    Args:
        actions: shape (N, 14)
        intervention: shape (N,) or None — 逐帧 intervention flag
        subset: "dagger" | "base" | "inference"

    Returns:
        shape (N,) int8 array of class labels
    """
    n = len(actions)
    arm_vel = compute_arm_velocity(actions)
    grip_vel = compute_gripper_velocity(actions)

    # ── 情况 1: base 示范 → 全部 class 5 ──
    if subset == "base":
        return np.full(n, CLASS_DEMO, dtype=np.int8)

    # ── 情况 2: inference (策略自主) → 全部 class 0, 但检测尾部静止 ──
    if subset == "inference":
        classes = np.full(n, CLASS_ROBOT, dtype=np.int8)
        # 从末尾向前扫: 标记静止尾巴
        for i in range(n - 1, -1, -1):
            if arm_vel[i] < STATIONARY_THR and grip_vel[i] < 0.02:
                classes[i] = CLASS_STATIONARY_TAIL
            else:
                break
        return classes

    # ── 情况 3: dagger (人类遥操) — 主要逻辑 ──
    classes = np.full(n, CLASS_INTV_CORE, dtype=np.int8)

    # Step A: 找 intervention 切换点
    transitions = []
    has_intervention_col = intervention is not None
    if has_intervention_col:
        transitions = find_intervention_transitions(intervention)

    # Step B: 对每个 human segment 做 hesitation + stationary 检测
    if has_intervention_col and len(transitions) > 0:
        # 有切换点的 episode: 处理每个 human segment
        # 找到所有 [takeover_frame, release_frame) 的 human segment
        human_segments = []
        in_human = False
        seg_start = 0
        for i in range(n):
            iv = int(intervention[i])
            if iv == 1 and not in_human:
                seg_start = i
                in_human = True
            elif iv != 1 and in_human:
                human_segments.append((seg_start, i))
                in_human = False
        if in_human:
            human_segments.append((seg_start, n))

        # 标记 preintv: 每个 takeover 前的 PREINTV_MARGIN 帧
        for t in transitions:
            if t["type"] == "takeover":
                pre_start = max(0, t["frame"] - PREINTV_MARGIN)
                classes[pre_start:t["frame"]] = CLASS_PREINTV

        # 对每个 human segment: 第一个 segment 做 hesitation 检测 (开头=接管),
        # 后续 segment 也做 (回到 human 控制=另一次接管)
        for i, (seg_start, seg_end) in enumerate(human_segments):
            # 每个 human segment 开头都是一次接管 → 做迟疑检测
            _mark_human_segment(classes, arm_vel, grip_vel, seg_start, seg_end,
                                take_over=True)
    else:
        # 纯遥操 episode (全部帧 human, 无切换点).
        # 对于 dagger 采集, episode 开始 = 操作员接管 → 也应做迟疑检测.
        _mark_human_segment(classes, arm_vel, grip_vel, 0, n, take_over=True)

    return classes


def _mark_human_segment(
    classes: np.ndarray,
    arm_vel: np.ndarray,
    grip_vel: np.ndarray,
    seg_start: int,
    seg_end: int,
    take_over: bool,
) -> None:
    """对一段 human teleop segment 标记 hesitation 和 stationary_tail.

    只有 seg_start 在 episode 开头 或 紧跟 intervention 切换 时才做 hesitation
    检测 (take_over=True)。中间的 human segment 不做 hesitation 检测。
    """
    # ── 前段迟疑检测: episode 开头或 takeover 后的低速起手 ──
    # 需要 velocity 连续 HESITATION_WIN 帧超过 HESITATION_THR 才结束迟疑
    if take_over:
        check_range = min(seg_end - seg_start, HESITATION_MAX)
        burst_count = 0
        hesitation_end = seg_start
        for i in range(seg_start, seg_start + check_range):
            if arm_vel[i] > HESITATION_THR:
                burst_count += 1
                if burst_count >= HESITATION_WIN:
                    hesitation_end = i - HESITATION_WIN + 1
                    break
            else:
                burst_count = 0
        # 标记迟疑帧
        if hesitation_end > seg_start:
            classes[seg_start:hesitation_end] = CLASS_HESITATION

    # ── 尾段静止检测: 从末尾向前扫 ──
    for i in range(seg_end - 1, seg_start - 1, -1):
        if arm_vel[i] < STATIONARY_THR and grip_vel[i] < 0.02:
            # 只覆盖还没被 hesitation 标记的帧
            if classes[i] == CLASS_INTV_CORE:
                classes[i] = CLASS_STATIONARY_TAIL
        else:
            break


def process_parquet(
    pq_path: str,
    subset: str,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    """处理单个 parquet 文件, 添加 dagger_frame_class 列.

    Returns:
        dict: {"path": str, "n_frames": int, "class_counts": dict, "episode_index": int}
    """
    table = pq.read_table(pq_path)
    n = table.num_rows
    ep_idx = table.column("episode_index")[0].as_py() if n > 0 else -1

    # 读取 action
    actions = np.array([table.column("action")[i].as_py() for i in range(n)], dtype=np.float64)

    # 读取 intervention (可能不存在)
    intervention = None
    if "intervention" in table.column_names:
        intervention = np.array(table.column("intervention").to_pylist(), dtype=np.int8)

    # 分类
    classes = classify_frames(actions, intervention, subset)

    # 统计
    class_counts = {int(c): int(np.sum(classes == c)) for c in range(6) if np.sum(classes == c) > 0}

    if dry_run:
        return {
            "path": pq_path,
            "n_frames": n,
            "episode_index": ep_idx,
            "class_counts": class_counts,
            "classes": classes,
        }

    # 备份原文件
    if backup:
        backup_path = pq_path + ".bak"
        if not os.path.exists(backup_path):
            shutil.copy2(pq_path, backup_path)

    # 添加新列
    new_col = pa.array(classes, type=pa.int8())
    # PyArrow 不支持原地添加列, 需要新建 table
    new_table = table.append_column("dagger_frame_class", new_col)
    pq.write_table(new_table, pq_path)

    return {
        "path": pq_path,
        "n_frames": n,
        "episode_index": ep_idx,
        "class_counts": class_counts,
    }


def process_dataset(
    dataset_path: str,
    subset: str,
    dry_run: bool = False,
    backup: bool = True,
    episodes: list[int] | None = None,
) -> list[dict]:
    """处理整个数据集目录下的所有 parquet 文件.

    Args:
        dataset_path: 数据集根目录 (含 data/chunk-*/)
        subset: "dagger" | "base" | "inference" | "auto"
        episodes: 可选, 只处理指定的 episode index 列表
    """
    pattern = os.path.join(dataset_path, "data", "chunk-*", "*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] 没找到 parquet 文件: {pattern}")
        return []

    # 自动检测 subset
    if subset == "auto":
        subset = _detect_subset(dataset_path, files)

    print(f"[INFO] subset={subset}, 找到 {len(files)} 个 parquet 文件")
    if dry_run:
        print("[DRY-RUN] 预览模式, 不改文件")
    if backup and not dry_run:
        print("[INFO] 备份模式: 每个 parquet 会生成 .bak 备份")

    results = []
    total_class_counts = {i: 0 for i in range(6)}

    for f in files:
        # 提取 episode index
        basename = os.path.basename(f)
        ep_idx = int(basename.replace("episode_", "").replace(".parquet", ""))
        if episodes is not None and ep_idx not in episodes:
            continue

        try:
            r = process_parquet(f, subset, dry_run=dry_run, backup=backup)
            results.append(r)
            for c, cnt in r["class_counts"].items():
                total_class_counts[c] += cnt
        except Exception as e:
            print(f"[ERROR] {f}: {e}", file=sys.stderr)
            continue

    # 汇总报告
    print("\n" + "=" * 60)
    print(f"处理完成: {len(results)} episodes, {sum(r['n_frames'] for r in results)} 总帧")
    print("-" * 60)
    for c in range(6):
        if total_class_counts[c] > 0:
            pct = total_class_counts[c] / max(1, sum(total_class_counts.values())) * 100
            print(f"  class {c} ({CLASS_NAMES[c]:20s}): {total_class_counts[c]:8d} 帧 ({pct:5.1f}%)")
    print("=" * 60)

    return results


def _detect_subset(dataset_path: str, files: list[str]) -> str:
    """从路径和数据内容自动检测 subset 类型."""
    path_lower = dataset_path.lower()
    if "/base/" in path_lower or "/demo/" in path_lower:
        return "base"
    if "/inference/" in path_lower or "/inf/" in path_lower:
        return "inference"
    if "/dagger/" in path_lower:
        return "dagger"

    # 从第一个文件的 intervention 列判断
    if files:
        t = pq.read_table(files[0])
        if "intervention" in t.column_names:
            ivs = set(t.column("intervention").to_pylist())
            if ivs == {0}:
                return "inference"
            if ivs == {1}:
                return "dagger"
            if -1 in ivs and 0 not in ivs and 1 not in ivs:
                return "base"
    return "dagger"  # 默认


def plot_episode(pq_path: str, output_png: str | None = None) -> None:
    """可视化单个 episode 的 velocity + 分类结果. 需要 matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] matplotlib 未安装, 无法绘图. pip install matplotlib")
        return

    table = pq.read_table(pq_path)
    n = table.num_rows
    actions = np.array([table.column("action")[i].as_py() for i in range(n)], dtype=np.float64)

    # 检查是否有已分类的列
    if "dagger_frame_class" in table.column_names:
        classes = np.array(table.column("dagger_frame_class").to_pylist(), dtype=np.int8)
    else:
        intervention = None
        if "intervention" in table.column_names:
            intervention = np.array(table.column("intervention").to_pylist(), dtype=np.int8)
        subset = _detect_subset(str(Path(pq_path).parent.parent.parent), [pq_path])
        classes = classify_frames(actions, intervention, subset)

    arm_vel = compute_arm_velocity(actions)
    grip_vel = compute_gripper_velocity(actions)

    colors = {
        0: "#1f77b4",  # robot - blue
        1: "#2ca02c",  # intv_core - green
        2: "#ff7f0e",  # preintv - orange
        3: "#d62728",  # hesitation - red
        4: "#9467bd",  # stationary_tail - purple
        5: "#7f7f7f",  # demo - gray
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 上: velocity + 分类色带
    ax1.plot(arm_vel, color="black", alpha=0.5, lw=0.5, label="arm velocity (mean |Δ|)")
    ax1.axhline(HESITATION_THR, color="red", ls="--", lw=0.8, label=f"hesitation thr={HESITATION_THR}")
    ax1.axhline(STATIONARY_THR, color="purple", ls=":", lw=0.8, label=f"stationary thr={STATIONARY_THR}")
    # 给背景按 class 着色
    prev_c = classes[0]
    seg_start = 0
    for i in range(1, n):
        if classes[i] != prev_c or i == n - 1:
            ax1.axvspan(seg_start, i, alpha=0.15, color=colors.get(int(prev_c), "gray"))
            seg_start = i
            prev_c = classes[i]
    ax1.set_ylabel("arm velocity (rad/frame)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.set_title(f"Episode {Path(pq_path).name} — frame classification")

    # 下: gripper velocity
    ax2.plot(grip_vel, color="green", alpha=0.6, lw=0.5, label="gripper velocity (max |Δ|)")
    ax2.set_ylabel("gripper velocity")
    ax2.set_xlabel("frame index")
    ax2.legend(fontsize=7)

    # 图例
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=colors[c], alpha=0.5, label=f"{c}: {CLASS_NAMES[c]}")
                      for c in sorted(set(int(c) for c in classes))]
    ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][:3],
               fontsize=6, loc="upper right", ncol=2)

    fig.tight_layout()
    if output_png:
        fig.savefig(output_png, dpi=150)
        print(f"[INFO] 图表保存到 {output_png}")
    else:
        fig.savefig("/tmp/classify_dagger_frames_debug.png", dpi=150)
        print("[INFO] 图表保存到 /tmp/classify_dagger_frames_debug.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="为 dagger 数据集逐帧打 dagger_frame_class 标签",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset_path", nargs="?", default=None,
                        help="数据集根目录 (含 data/chunk-*/). --plot 模式下可选")
    parser.add_argument("--subset", default="auto",
                        choices=["auto", "dagger", "base", "inference"],
                        help="数据集类型 (默认 auto=自动检测)")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式: 只统计不改文件")
    parser.add_argument("--no-backup", action="store_true",
                        help="不生成 .bak 备份 (危险)")
    parser.add_argument("--episodes", type=str, default=None,
                        help="只处理指定 episode, 逗号分隔 (如 0,1,5-10)")
    parser.add_argument("--plot", type=str, default=None,
                        help="可视化单个 episode 的 velocity+分类 (需要 matplotlib), "
                             "参数为 parquet 文件路径")
    parser.add_argument("--plot-output", type=str, default=None,
                        help="--plot 的输出 PNG 路径")
    args = parser.parse_args()

    # 可视化模式
    if args.plot:
        plot_episode(args.plot, args.plot_output)
        return

    if not args.dataset_path:
        parser.error("dataset_path 必须提供 (--plot 模式除外)")

    # 解析 episodes
    episodes = None
    if args.episodes:
        episodes = set()
        for part in args.episodes.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                episodes.update(range(int(a), int(b) + 1))
            else:
                episodes.add(int(part))
        episodes = sorted(episodes)

    process_dataset(
        args.dataset_path,
        subset=args.subset,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        episodes=episodes,
    )


if __name__ == "__main__":
    main()

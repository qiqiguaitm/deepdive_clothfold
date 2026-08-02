#!/usr/bin/env python
"""[PR-2 显著性管线] RoboTwin 2ckpt×Nseed×难度 eval 聚合 + Welch t。

修复原 eval yaml 内嵌聚合的坑: group 名 hard 版带 `_hard` 后缀 → 原 glob 用 base 名 → n=0 假空。
本脚本自动同时收 Easy(rt_*_balanced)+ Hard(rt_*_balanced_hard), 逐任务 baseline vs LMWM 均值±SE + Welch t + 净Δ。
结果 json 键: success_rate(每任务每seed一个)。arm 从路径 rt_lmwm/rt_baseline 判, 难度从 _hard 判, 任务从路径任务名判。

用法:
  python rt_eval_significance.py --root <eval_runs/robotwin> [--tasks t1 t2 ...]
方法论纪律(对比 Spatial Forcing 无seed/无误差棒): 多 seed + Welch t + 明确 n。
"""
import json, glob, re, argparse, statistics as st
from collections import defaultdict

TASKS = ["beat_block_hammer", "handover_block", "blocks_ranking_rgb",
         "blocks_ranking_size", "stack_blocks_two", "stack_blocks_three"]


def welch(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None, None
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = (va / na + vb / nb) ** 0.5
    if se == 0:
        return (float("inf") if ma != mb else 0.0), None
    t = (mb - ma) / se
    denom = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = (va / na + vb / nb) ** 2 / denom if denom > 0 else na + nb - 2
    return t, df


def collect(root):
    # (diff, arm, task) -> {seed: SR%}
    cell = defaultdict(dict)
    for f in glob.glob(f"{root}/rt_*balanced*/seed*/**/*.json", recursive=True):
        if "task_status" in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or "success_rate" not in d:
            continue
        arm = "LMWM" if "rt_lmwm" in f else ("baseline" if "rt_baseline" in f else "?")
        diff = "Hard" if "balanced_hard" in f else "Easy"
        task = next((t for t in TASKS if t in f), "?")
        m = re.search(r"/seed(\d+)/", f)
        seed = int(m.group(1)) if m else -1
        cell[(diff, arm, task)][seed] = 100 * d["success_rate"]
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--sig-t", type=float, default=2.4, help="|t| 阈值(近似 p<0.05, df~3-6)")
    a = ap.parse_args()
    cell = collect(a.root)
    for diff in ["Easy", "Hard"]:
        print(f"\n===== RoboTwin balanced {diff} — LMWM vs baseline (Welch t) =====")
        print(f"{'task':22s}{'baseline':>14s}{'LMWM':>14s}{'Δ':>7s}{'t':>7s}{'sig':>6s}")
        nets = []
        for t in a.tasks:
            A = cell.get((diff, "baseline", t), {})
            B = cell.get((diff, "LMWM", t), {})
            av = [A[s] for s in sorted(A)]
            bv = [B[s] for s in sorted(B)]
            if not av or not bv:
                print(f"{t:22s}{'(缺)':>14s}")
                continue
            tv, df = welch(av, bv)
            sig = "★" if tv is not None and abs(tv) > a.sig_t else ""
            fa = f"{st.mean(av):.1f}±{(st.pstdev(av)/len(av)**.5):.1f}(n{len(av)})"
            fb = f"{st.mean(bv):.1f}±{(st.pstdev(bv)/len(bv)**.5):.1f}(n{len(bv)})"
            d = st.mean(bv) - st.mean(av)
            nets.append(d)
            print(f"{t:22s}{fa:>14s}{fb:>14s}{d:>+7.1f}{(tv or 0):>7.2f}{sig:>6s}")
        if nets:
            print(f"  {'6任务净Δ均值':22s}{'':>28s}{st.mean(nets):>+7.2f}  "
                  f"({'LMWM净赢' if st.mean(nets) > 0 else 'LMWM净负'})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""[B step2·均衡版] robotwin milestone pairs —— p1_robotwin_rvalley_pairs.py 的分组修复版。

原版 bug: 按 robotwin `task_index`(**语言指令增强级**, 每变体 <5ep)分组 → MIN_EP=5 把
  hammer/stack 几乎全过滤(见 RESULTS_robotwin_P2 附2.2)。本版改为**按规范任务分组**
  (hammer/stack_two/stack_three/ranking_rgb/ranking_size/handover), 每类几百 ep, r-field 有
  足够 cross-ep 复现算 canonical milestone。只处理 6 积木任务(跳过其它 robotwin 任务)。

r-field 分段配方与原版逐字一致(THR/平滑/find_peaks/脊选取)。特征源 robotwin_dinov3base(pooled)。
pair_task = 规范任务 id(0-5, 见 CANON_ID)。输出 robotwin_milestone_balanced/pairs.npz。
用法: python p1_robotwin_rvalley_pairs_balanced.py
"""
import argparse
import os, glob, json
import numpy as np, pyarrow.parquet as pq, torch
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict

_DEV = "cuda" if torch.cuda.is_available() else "cpu"

REPO = "/vePFS/tim/workspace/deepdive_kai0"
ROOT = f"{REPO}/lmvla/lawam/dataset/robotwin2.0"
FEAT = f"{REPO}/lmvla/lmwm/data/robotwin_dinov3base"
OUT = f"{REPO}/lmvla/lmwm/data/robotwin_milestone_balanced"
THR = 0.03
MIN_EP = 5
# 每规范组子采样上限: 大组(694ep)全帧 cdist(N,N) 会到 ~43GB/极慢; 200ep 远超估 canonical
# milestone 所需(原按 task_index 分组常<50), 且够训练。均匀采(确定性, 可复现)。
CAP = 200

CANON_ID = {"hammer": 0, "stack_two": 1, "stack_three": 2,
            "ranking_rgb": 3, "ranking_size": 4, "handover": 5}

# RoboTwin 2.0 stores 50 official tasks as contiguous 550-episode blocks in
# DEFAULT_TASKS order. Use those stable block IDs instead of augmented language:
# language matching previously mixed stack_bowls into stack_blocks and treated
# unrelated "right arm" instructions as handover.
OFFICIAL_TASK_BLOCK = {
    "hammer": 1,
    "ranking_rgb": 2,
    "ranking_size": 3,
    "handover": 8,
    "stack_three": 44,
    "stack_two": 45,
}
EPISODES_PER_TASK = 550


def canon(s):
    z = str(s).lower()
    if "hammer" in z or "beat the block" in z: return "hammer"
    if "bell" in z or "click" in z or "press" in z or "alarm" in z: return None  # 非积木, 排除
    if "stack" in z or "on top of" in z:
        return "stack_three" if ("three" in z or ("small block" in z and "medium block" in z and "large block" in z)) else "stack_two"
    rgb = ("red block" in z and "green block" in z and "blue block" in z)
    size = ("large block" in z and "medium block" in z and "small block" in z)
    if rgb and not size: return "ranking_rgb"
    if size and not rgb: return "ranking_size"
    if any(w in z for w in ["hand", "pass", "transfer", "pad", "right arm"]): return "handover"
    return None  # 其它 robotwin 任务, 排除


def l2(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def r_and_segments(gd):
    """逐字同 LIBERO/原版: 每 ep r[n], 段边界 seg, 段脊 ridge。"""
    eps = list(gd); F = l2(np.concatenate([gd[e] for e in eps]).astype(np.float32))
    ep = np.concatenate([np.full(len(gd[e]), i) for i, e in enumerate(eps)]); ne = len(eps)
    lens = [len(gd[e]) for e in eps]; offs = np.cumsum([0] + lens)
    # GPU cdist(RoboTwin ep 长~144, 单线程 cdist 太慢; A100 秒级)
    Ft = torch.from_numpy(F).to(_DEV)
    ep_t = torch.from_numpy(ep).to(_DEV)
    D = torch.cdist(Ft, Ft)                                  # [Ntot,Ntot]
    dmin = torch.full((len(F), ne), 1e9, device=_DEV)
    for j in range(ne): dmin[:, j] = D[:, ep_t == j].min(1).values
    dmin = dmin.cpu().numpy().astype(np.float32)
    del D, Ft, ep_t
    if _DEV == "cuda": torch.cuda.empty_cache()
    other = ep[:, None] != np.arange(ne)[None]; sig = np.median(dmin[other])
    r = (np.exp(-dmin**2 / (2 * sig * sig)) * other).sum(1) / (ne - 1)
    res = {}
    for i, e in enumerate(eps):
        s, en = offs[i], offs[i + 1]; n = en - s; rr = r[s:en]
        v, _ = find_peaks(-gaussian_filter1d(rr, 1.4), prominence=THR, distance=max(2, n // 12))
        seg = [0] + list(v) + [n]; ridge = [a + int(np.argmax(rr[a:b])) for a, b in zip(seg[:-1], seg[1:])]
        res[e] = (seg, ridge, n)
    return res


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--feat", default=FEAT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument(
        "--official-blocks",
        action="store_true",
        help="Select the six official RoboTwin task blocks. Required for clean all6 evidence.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root, feat, out, cap = args.root, args.feat, args.out, args.cap
    files = sorted(glob.glob(f"{feat}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    cached = [int(os.path.basename(f)[2:-4]) for f in files]
    print(f"[feat] {len(cached)} ep 有 pooled 特征", flush=True)

    task_eps = defaultdict(list)
    if args.official_blocks:
        cached_set = set(cached)
        for name, block in OFFICIAL_TASK_BLOCK.items():
            lo = block * EPISODES_PER_TASK
            hi = lo + EPISODES_PER_TASK
            task_eps[name] = [e for e in range(lo, hi) if e in cached_set]
    else:
        ep2str = {}
        for line in open(f"{root}/meta/episodes.jsonl"):
            d = json.loads(line)
            t = d.get("tasks")
            ep2str[d["episode_index"]] = t[0] if isinstance(t, list) else t
        for e in cached:
            c = canon(ep2str.get(e, ""))
            if c is not None:
                task_eps[c].append(e)
    print("[规范任务分布(有特征)]", {k: len(v) for k, v in sorted(task_eps.items())}, flush=True)

    cur_ep, cur_fi, tgt_fi, cur_ms, pair_task = [], [], [], [], []
    nseg = []
    used_eps = 0
    selected_by_task = {}
    for c, teps in sorted(task_eps.items()):
        if len(teps) < MIN_EP:
            print(f"  ! {c}: {len(teps)}ep < {MIN_EP}, 跳过", flush=True); continue
        teps = sorted(teps)
        if len(teps) > cap:
            idx = np.linspace(0, len(teps) - 1, cap).round().astype(int)
            teps = sorted(set(teps[i] for i in idx))
            print(f"  · {c}: 子采样 -> {len(teps)}ep (原组>CAP={cap})", flush=True)
        selected_by_task[c] = teps
        # Load only the selected task block. Loading all 27,500 episodes at once
        # can consume hundreds of GiB before the first cdist.
        group_gist = {
            e: np.load(f"{feat}/ep{e}.npz")["pooled"].astype(np.float32)
            for e in teps
        }
        res = r_and_segments(group_gist)
        tid = CANON_ID[c]
        seglens = []
        for e in teps:
            seg, ridge, n = res[e]; nseg.append(len(ridge)); seglens.append(len(ridge)); used_eps += 1
            for p in range(n):
                si = np.searchsorted(seg, p, "right") - 1
                tgt = ridge[si + 1] if si + 1 < len(ridge) else n - 1
                cur_ep.append(e); cur_fi.append(p); tgt_fi.append(tgt); cur_ms.append(si); pair_task.append(tid)
        print(f"  {c:12s}(id{tid}): {len(teps)}ep, 每ep段数中位={int(np.median(seglens))} 范围[{min(seglens)},{max(seglens)}]", flush=True)
        del group_gist

    arrs = [np.array(x) for x in (cur_ep, cur_fi, tgt_fi, cur_ms, pair_task)]
    print(f"[cover] 建对 {used_eps} eps, {len(set(pair_task))} 规范任务", flush=True)
    print(f"[pairs] {len(cur_ep)} 对", flush=True)
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/pairs.npz", cur_ep=arrs[0], cur_fi=arrs[1], tgt_fi=arrs[2], cur_ms=arrs[3], pair_task=arrs[4])
    json.dump({v: k for k, v in CANON_ID.items()}, open(f"{out}/canon_id.json", "w"))
    json.dump(
        {
            "selection": "official_task_blocks" if args.official_blocks else "legacy_language_match",
            "episodes_per_task": EPISODES_PER_TASK,
            "cap": cap,
            "official_task_block": OFFICIAL_TASK_BLOCK,
            "selected_episode_count": {k: len(v) for k, v in selected_by_task.items()},
            "selected_episode_range": {k: [min(v), max(v)] for k, v in selected_by_task.items()},
        },
        open(f"{out}/selection_manifest.json", "w"),
        indent=2,
        sort_keys=True,
    )
    print(f"[save] {out}/pairs.npz\nDONE_BALANCED", flush=True)


if __name__ == "__main__":
    main()

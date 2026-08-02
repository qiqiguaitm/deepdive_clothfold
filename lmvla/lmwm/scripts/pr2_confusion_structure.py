#!/usr/bin/env python
"""PR-2 裁决实验: 跨-ep 检索混淆分数 + 结构度分数, 对预注册预测①② (PAPER_PLAN §16).

预测①: spatial "on the ramekin" (语义 t5) 混淆分数=spatial组极端离群, 主要混向
        "between the plate and ramekin"/"next to the ramekin".
预测②: 40 任务逐任务 Δ(dual2q−nowm) 与混淆负相关、与结构度正相关 (Spearman).

数据:
  特征  lmvla/lmwm/data/pi05_feat/libero_v2p1_dinov3base/<suite>/ep*.npz  key grid[N,256,768] fp16
  ep→task  /vePFS/tim/workspace/LIBERO_fastwam/<suite>_no_noops_lerobot parquet (episode_index,task_index)
  Δ    libero_10: 真 per-episode episodes.jsonl (nowm vs dual2q); 其余套件仅 doc 聚合(见报告).

用法: kai0/.venv/bin/python lmvla/lmwm/scripts/pr2_confusion_structure.py [--frames_per_ep 8]
"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd

ROOT = "/vePFS/tim/workspace/deepdive_kai0"
FEAT = f"{ROOT}/lmvla/lmwm/data/pi05_feat/libero_v2p1_dinov3base"
LEROBOT = "/vePFS/tim/workspace/LIBERO_fastwam/{s}_no_noops_lerobot"
CRAVE_SRC = f"{ROOT}/lmvla/crave/src"
EVAL = f"{ROOT}/lmvla/lawam/results/eval_runs/libero"
NOWM = f"{EVAL}/20260718_211747_lawam_nowm_cnsh_volc/20260719_032133/suites/libero_10/episodes.jsonl"
DUAL2Q = f"{EVAL}/dual2q_cfg15/20260718_212329/suites/libero_10/episodes.jsonl"
DUAL2Q_REVAL = sorted(glob.glob(f"{EVAL}/revalidate_dual2q/seed*/suites/libero_10/episodes.jsonl"))
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
OUT_NPZ = f"{ROOT}/lmvla/lmwm/data/pr2_scores.npz"


def load_suite(s):
    """返回 dict[ep] = gist[N,768] fp32, 以及 ep2task, task2desc."""
    files = sorted(glob.glob(f"{FEAT}/{s}/ep*.npz"), key=lambda p: int(os.path.basename(p)[2:-4]))
    gist = {}
    for f in files:
        e = int(os.path.basename(f)[2:-4])
        g = np.load(f)["grid"]  # [N,256,768] fp16
        gist[e] = g.astype(np.float32).mean(1)  # [N,768]
    root = LEROBOT.format(s=s)
    pq = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
    dd = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"]) for p in pq])
    ep2task = dd.groupby("episode_index")["task_index"].first().to_dict()
    task2desc = {json.loads(l)["task_index"]: json.loads(l)["task"]
                 for l in open(f"{root}/meta/tasks.jsonl")}
    return gist, ep2task, task2desc


def per_task_sr(jsonl):
    """episodes.jsonl -> {task_description: SR}."""
    succ, tot, desc = {}, {}, {}
    for l in open(jsonl):
        d = json.loads(l)
        t = d["task_id"]
        desc[t] = d["task_description"]
        succ[t] = succ.get(t, 0) + int(bool(d["success"]))
        tot[t] = tot.get(t, 0) + 1
    return {desc[t]: 100.0 * succ[t] / tot[t] for t in tot}, {desc[t]: tot[t] for t in tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_per_ep", type=int, default=8)
    ap.add_argument("--max_frames_per_task", type=int, default=2000)
    args = ap.parse_args()
    sys.path.insert(0, CRAVE_SRC)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. 加载全部 4 套件特征 + 映射 ----
    suite_data = {}
    global_tasks = []   # 列表 of (suite, task_index, desc)
    for s in SUITES:
        gist, ep2task, task2desc = load_suite(s)
        suite_data[s] = (gist, ep2task, task2desc)
        for ti in sorted(task2desc):
            global_tasks.append((s, ti, task2desc[ti]))
        print(f"[load] {s}: {len(gist)} eps, {sum(len(v) for v in gist.values())} frames, {len(task2desc)} tasks", flush=True)
    T = len(global_tasks)
    gt_index = {(s, ti): k for k, (s, ti, _) in enumerate(global_tasks)}
    print(f"[tasks] 全局 {T} 任务", flush=True)

    # ---- 2. 采样帧建 bank (跨全部 40 任务) ----
    feats, f_task, f_ep = [], [], []   # f_ep 用全局唯一 ep id 防跨套件同号
    ep_uid = 0
    ep_uid_map = {}
    rng = np.random.default_rng(0)
    for s in SUITES:
        gist, ep2task, task2desc = suite_data[s]
        # 每任务累计帧上限
        task_cnt = {}
        for e in sorted(gist):
            ti = ep2task.get(e, -1)
            if ti < 0:
                continue
            k = gt_index[(s, ti)]
            g = gist[e]
            n = len(g)
            m = min(args.frames_per_ep, n)
            idx = np.linspace(0, n - 1, m).round().astype(int)
            if task_cnt.get(k, 0) >= args.max_frames_per_task:
                continue
            uid = ep_uid; ep_uid_map[(s, e)] = uid; ep_uid += 1
            for j in idx:
                feats.append(g[j]); f_task.append(k); f_ep.append(uid)
            task_cnt[k] = task_cnt.get(k, 0) + len(idx)
    feats = np.stack(feats).astype(np.float32)
    f_task = np.array(f_task); f_ep = np.array(f_ep)
    # L2 normalize
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    print(f"[bank] {len(feats)} 帧, {ep_uid} eps", flush=True)

    # ---- 3. 跨-ep 最近邻 (排除本 ep) -> 混淆 ----
    Xb = torch.from_numpy(feats).to(dev)      # [Nf,768] normalized
    ep_t = torch.from_numpy(f_ep).to(dev)
    tsk_t = torch.from_numpy(f_task).to(dev)
    Nf = len(feats)
    nn_task = np.empty(Nf, dtype=np.int64)
    B = 2048
    for i in range(0, Nf, B):
        q = Xb[i:i + B]                        # [b,768]
        sim = q @ Xb.T                         # [b,Nf] cosine
        same_ep = ep_t[i:i + B, None] == ep_t[None, :]
        sim = sim.masked_fill(same_ep, -2.0)   # 排除同 ep 帧(含自身)
        nn = sim.argmax(1)
        nn_task[i:i + B] = tsk_t[nn].cpu().numpy()
    # per-task 混淆率 + 40x40 矩阵(行=query task, 列=NN task 分布)
    conf_rate = np.zeros(T); conf_mat = np.zeros((T, T))
    for k in range(T):
        m = f_task == k
        if m.sum() == 0:
            continue
        nnk = nn_task[m]
        conf_rate[k] = float(np.mean(nnk != k))
        for j in range(T):
            conf_mat[k, j] = float(np.mean(nnk == j))
    print("[confusion] done", flush=True)

    # ---- 4. 结构度: per-task CRAVE milestone 数 M + 每 ep 平均段数 ----
    from crave.clustering import build_clusters
    struct_M = np.full(T, np.nan); struct_seg = np.full(T, np.nan)
    for s in SUITES:
        gist, ep2task, task2desc = suite_data[s]
        from collections import defaultdict
        task_eps = defaultdict(list)
        for e in sorted(gist):
            ti = ep2task.get(e, -1)
            if ti >= 0:
                task_eps[ti].append(e)
        for ti, teps in task_eps.items():
            k = gt_index[(s, ti)]
            if len(teps) < 5:
                struct_M[k] = 0; struct_seg[k] = 0; continue
            F = np.concatenate([gist[e] for e in teps])
            E = np.concatenate([np.full(len(gist[e]), e) for e in teps])
            Tv = np.concatenate([np.linspace(0, 1, len(gist[e])) for e in teps])
            cl = build_clusters(F, E, Tv, len(teps), seed=0)
            C = cl["C"]; M = cl["M"]
            struct_M[k] = M
            # 每 ep 平均 milestone 段数(平滑+单调后 distinct 值数)
            segs = []
            for e in teps:
                ge = gist[e]
                raw = np.linalg.norm(ge[:, None] - C[None], axis=2).argmin(1)
                w = 5; sm = raw.copy()
                for i in range(len(raw)):
                    sm[i] = int(np.median(raw[max(0, i - w): i + w + 1]))
                sm = np.maximum.accumulate(sm)
                segs.append(len(np.unique(sm)))
            struct_seg[k] = float(np.mean(segs))
        print(f"[structure] {s} done", flush=True)

    # ---- 5. Δ(dual2q − nowm) ----
    nowm_sr, nowm_n = per_task_sr(NOWM)
    d2q_sr, d2q_n = per_task_sr(DUAL2Q)
    # revalidate 平均(鲁棒性)
    reval = [per_task_sr(p)[0] for p in DUAL2Q_REVAL]
    delta = np.full(T, np.nan); delta_src = ["" for _ in range(T)]
    nowm_col = np.full(T, np.nan); d2q_col = np.full(T, np.nan)
    # libero_10 真数(按 description join)
    for k, (s, ti, desc) in enumerate(global_tasks):
        if s == "libero_10" and desc in nowm_sr and desc in d2q_sr:
            nowm_col[k] = nowm_sr[desc]; d2q_col[k] = d2q_sr[desc]
            delta[k] = d2q_sr[desc] - nowm_sr[desc]; delta_src[k] = "libero10_real"
    # revalidate dual2q 平均版 Δ(可选鲁棒)
    d2q_reval_mean = {}
    if reval:
        allk = set().union(*[set(r) for r in reval])
        for kk in allk:
            vals = [r[kk] for r in reval if kk in r]
            d2q_reval_mean[kk] = float(np.mean(vals))
    delta_reval = np.full(T, np.nan)
    for k, (s, ti, desc) in enumerate(global_tasks):
        if s == "libero_10" and desc in nowm_sr and desc in d2q_reval_mean:
            delta_reval[k] = d2q_reval_mean[desc] - nowm_sr[desc]

    # 其余套件: doc 聚合(RESULTS_libero_4suite) 重建, 明确标注为 doc 近似
    #   spatial: "on the ramekin" (语义 t5) nowm 76 -> dual2q 42 (Δ=-34); 其余 9 任务 97-100 -> Δ≈0
    #   goal: 聚合 Δ=0 (饱和) -> 全 0 ; object: 聚合 Δ=-0.3 (饱和) -> 全 ~0
    for k, (s, ti, desc) in enumerate(global_tasks):
        if s == "libero_spatial":
            if "on the ramekin" in desc and "next to" not in desc:  # "on the ramekin"
                delta[k] = 42.0 - 76.0; nowm_col[k] = 76.0; d2q_col[k] = 42.0
                delta_src[k] = "doc_spatial_t5"
            else:
                delta[k] = 0.0; delta_src[k] = "doc_spatial_sat~0"
        elif s == "libero_goal":
            delta[k] = 0.0; delta_src[k] = "doc_goal_sat~0"
        elif s == "libero_object":
            delta[k] = -0.3; delta_src[k] = "doc_object_sat~0"

    # ---- 6. 相关 + 输出 ----
    from scipy.stats import spearmanr
    descs = [d for _, _, d in global_tasks]
    suites_arr = [s for s, _, _ in global_tasks]

    def corr(mask, x, y):
        m = mask & np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return (np.nan, np.nan, int(m.sum()))
        r, p = spearmanr(x[m], y[m])
        return (float(r), float(p), int(m.sum()))

    all_mask = np.ones(T, bool)
    l10_mask = np.array([s == "libero_10" for s in suites_arr])

    res = {
        "spearman_delta_confusion_40": corr(all_mask, delta, conf_rate),
        "spearman_delta_structM_40": corr(all_mask, delta, struct_M),
        "spearman_delta_structseg_40": corr(all_mask, delta, struct_seg),
        "spearman_delta_confusion_l10": corr(l10_mask, delta, conf_rate),
        "spearman_delta_structM_l10": corr(l10_mask, delta, struct_M),
        "spearman_delta_structseg_l10": corr(l10_mask, delta, struct_seg),
        "spearman_deltareval_confusion_l10": corr(l10_mask, delta_reval, conf_rate),
    }

    # t5 离群检验(spatial 组)
    sp_mask = np.array([s == "libero_spatial" for s in suites_arr])
    sp_conf = conf_rate[sp_mask]
    sp_idx = np.where(sp_mask)[0]
    t5_k = [k for k in sp_idx if "on the ramekin" in descs[k] and "next to" not in descs[k]][0]
    t5_conf = conf_rate[t5_k]
    sp_mean = float(np.mean(sp_conf)); sp_std = float(np.std(sp_conf))
    t5_z = (t5_conf - sp_mean) / (sp_std + 1e-9)
    t5_rank = int(np.sum(sp_conf > t5_conf))  # 0 = 最高
    # t5 主要混向谁
    t5_row = conf_mat[t5_k].copy(); t5_row[t5_k] = 0
    top_mix = np.argsort(-t5_row)[:6]

    np.savez(OUT_NPZ,
             global_tasks=np.array([f"{s}|t{ti}|{d}" for s, ti, d in global_tasks], dtype=object),
             suites=np.array(suites_arr, dtype=object),
             descs=np.array(descs, dtype=object),
             conf_rate=conf_rate, conf_mat=conf_mat,
             struct_M=struct_M, struct_seg=struct_seg,
             delta=delta, delta_reval=delta_reval,
             delta_src=np.array(delta_src, dtype=object),
             nowm_sr=nowm_col, dual2q_sr=d2q_col)

    # 打印摘要
    print("\n" + "=" * 70)
    print("RESULTS")
    for k, v in res.items():
        print(f"  {k}: r={v[0]:.4f} p={v[1]:.4g} n={v[2]}")
    print(f"\n[t5 outlier] spatial 'on the ramekin' (LeRobot t{global_tasks[t5_k][1]}):")
    print(f"  conf_rate={t5_conf:.4f}  spatial_mean={sp_mean:.4f}±{sp_std:.4f}  z={t5_z:.2f}  rank={t5_rank}/10 (0=最高)")
    print(f"  spatial 各任务 conf: " + ", ".join(f"t{global_tasks[k][1]}={conf_rate[k]:.3f}" for k in sp_idx))
    print(f"  t5 主要混向:")
    for j in top_mix:
        print(f"    {suites_arr[j]}|t{global_tasks[j][1]} ({descs[j][:50]}): {t5_row[j]:.3f}")

    # 保存报告数据(供 md)
    import pickle
    with open("/vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/e56c875e-3983-4035-972c-e9cb06ca942f/scratchpad/pr2_report.pkl", "wb") as fh:
        pickle.dump(dict(res=res, global_tasks=global_tasks, conf_rate=conf_rate,
                         conf_mat=conf_mat, struct_M=struct_M, struct_seg=struct_seg,
                         delta=delta, delta_reval=delta_reval, delta_src=delta_src,
                         nowm_col=nowm_col, d2q_col=d2q_col, t5_k=t5_k, t5_conf=t5_conf,
                         sp_mean=sp_mean, sp_std=sp_std, t5_z=t5_z, t5_rank=t5_rank,
                         top_mix=top_mix, descs=descs, suites_arr=suites_arr), fh)
    print("\n[save]", OUT_NPZ, "+ /vePFS/tim/tmp/claude-1000/-vePFS-tim-workspace-deepdive-kai0/e56c875e-3983-4035-972c-e9cb06ca942f/scratchpad/pr2_report.pkl")
    print("DONE")


if __name__ == "__main__":
    main()

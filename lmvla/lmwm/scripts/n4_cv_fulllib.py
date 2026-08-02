#!/usr/bin/env python
"""N4: full-library recompute of residual-vs-absolute discriminability CV + 86% redundancy.

Original (2ep) claim (PROGRESS_pi05_lmwm_P1_2026-07-23 §2.1, Fig.2 make_figs.py):
  absolute target g_next is 86% current state (redundant); real signal in residual r = g_next - h_t.
  along-trajectory CV: residual [1.112,1.212] vs absolute [0.119,0.087]  (~10x).

Definitions (reconstructed from GT pairs, per handoff N4):
  For each episode, along its frame trajectory:
    h_t        = pooled(grid[cur_fi])                 # current state (mean over 256 tokens)
    g_next     = pooled(grid[tgt_fi])                 # absolute milestone target (r-ridge frame)
    r          = g_next - h_t                         # residual target
  along-trajectory CV(v) = std_t(||v_t||) / mean_t(||v_t||)   over the episode's frames.
  86% redundancy = mean_t cos^2(g_next, h_t)  (energy of absolute target explained by current state);
                   informative fraction = 1 - cos^2 = sin^2 (orthogonal residual energy).
  Also report ||r||^2/||g||^2 energy-ratio variant.

Data: pairs.npz (cur_ep,cur_fi,tgt_fi,pair_task) + libero_dinov3base/ep{cur_ep}.npz[grid].
"""
import os, glob, json
import numpy as np

REPO = "/vePFS/tim/workspace/deepdive_kai0"
FEAT = f"{REPO}/lmvla/lmwm/data/libero_dinov3base"
PAIRS = f"{REPO}/lmvla/lmwm/data/libero_rvalley/pairs.npz"
OUT = f"{REPO}/lmvla/lmwm/data/n4_cv_fulllib.npz"

# suite membership by task_index (40 LIBERO tasks: 10 per suite, order = spatial,object,goal,10)
# We do NOT need exact suite names for CV; report per-task and overall.

def main():
    z = np.load(PAIRS)
    cur_ep, cur_fi, tgt_fi, pair_task = z["cur_ep"], z["cur_fi"], z["tgt_fi"], z["pair_task"]
    eps = sorted(set(cur_ep.tolist()))
    print(f"[data] {len(eps)} episodes, {len(cur_ep)} frame-pairs, {len(set(pair_task.tolist()))} tasks", flush=True)

    rows = []  # per-episode: (ep, task, cv_abs, cv_resid, cos2_mean, energyratio_mean, nframes)
    for ei, e in enumerate(eps):
        m = cur_ep == e
        cfi = cur_fi[m]; tfi = tgt_fi[m]; tk = int(pair_task[m][0])
        order = np.argsort(cfi)
        cfi = cfi[order]; tfi = tfi[order]
        grid = np.load(f"{FEAT}/ep{e}.npz")["grid"].astype(np.float32)  # [N,256,768]
        pooled = grid.mean(1)  # [N,768]
        h = pooled[cfi]              # current state per frame
        g = pooled[tfi]             # absolute target per frame
        r = g - h                    # residual
        an = np.linalg.norm(g, axis=1)
        rn = np.linalg.norm(r, axis=1)
        hn = np.linalg.norm(h, axis=1)
        # variant A: norm-CV = std_t(||v_t||)/mean_t(||v_t||)
        cv_abs = an.std() / (an.mean() + 1e-12)
        cv_res = rn.std() / (rn.mean() + 1e-12)
        # variant B: vector-CV = ||std_t(v)||_2 / ||mean_t(v)||_2  (matches 2ep anchor scale)
        vcv_abs = np.linalg.norm(g.std(0)) / (np.linalg.norm(g.mean(0)) + 1e-12)
        vcv_res = np.linalg.norm(r.std(0)) / (np.linalg.norm(r.mean(0)) + 1e-12)
        cos = (g * h).sum(1) / (an * hn + 1e-12)
        cos2 = cos ** 2                       # redundant energy fraction per frame
        eratio = (rn ** 2) / (an ** 2 + 1e-12)  # ||r||^2/||g||^2 informative-energy variant
        rows.append((e, tk, cv_abs, cv_res, cos2.mean(), eratio.mean(), len(cfi), vcv_abs, vcv_res))
        if ei % 200 == 0:
            print(f"  ep{e} task{tk} n={len(cfi)} cv_abs={cv_abs:.3f} cv_res={cv_res:.3f} cos2={cos2.mean():.3f}", flush=True)

    rows = np.array(rows, dtype=np.float64)
    ep_arr, task_arr = rows[:, 0], rows[:, 1]
    cv_abs, cv_res, cos2, eratio, nfr = rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]
    vcv_abs, vcv_res = rows[:, 7], rows[:, 8]

    def stats(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)), std=float(x.std()),
                    p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                    min=float(x.min()), max=float(x.max()))

    ratio = cv_res / (cv_abs + 1e-12)
    vratio = vcv_res / (vcv_abs + 1e-12)
    summary = {
        "n_episodes": int(len(rows)),
        "n_frames_total": int(nfr.sum()),
        "_NOTE": "variantA normCV = std_t(||v||)/mean_t(||v||); variantB vectorCV = ||std_t(v)||/||mean_t(v)|| (matches 2ep anchor scale abs~0.12 resid~1.1)",
        "vectorCV_absolute": stats(vcv_abs),
        "vectorCV_residual": stats(vcv_res),
        "vectorCV_ratio_resid_over_abs_perEp": stats(vratio),
        "vectorCV_ratio_of_medians": float(np.median(vcv_res) / np.median(vcv_abs)),
        "cv_absolute": stats(cv_abs),
        "cv_residual": stats(cv_res),
        "cv_ratio_resid_over_abs_perEp": stats(ratio),
        "cv_ratio_of_medians": float(np.median(cv_res) / np.median(cv_abs)),
        "cv_ratio_of_means": float(cv_res.mean() / cv_abs.mean()),
        "redundancy_cos2_meanOverEps": float(cos2.mean()),   # "86% current state"
        "redundancy_cos2_median": float(np.median(cos2)),
        "informative_1minus_cos2_mean": float(1 - cos2.mean()),
        "energyratio_resid_over_abs_mean": float(eratio.mean()),  # ||r||^2/||g||^2
    }
    # per-task
    per_task = {}
    for tk in sorted(set(task_arr.astype(int).tolist())):
        mm = task_arr.astype(int) == tk
        per_task[int(tk)] = dict(n=int(mm.sum()),
                                 vcv_abs=float(np.median(vcv_abs[mm])),
                                 vcv_res=float(np.median(vcv_res[mm])),
                                 cv_abs=float(np.median(cv_abs[mm])),
                                 cv_res=float(np.median(cv_res[mm])),
                                 cos2=float(cos2[mm].mean()))
    summary["per_task_median"] = per_task

    print("\n===== N4 SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez(OUT, ep=ep_arr, task=task_arr, cv_abs=cv_abs, cv_res=cv_res,
             vcv_abs=vcv_abs, vcv_res=vcv_res, cos2=cos2, eratio=eratio, nframes=nfr)
    with open(OUT.replace(".npz", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()

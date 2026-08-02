#!/usr/bin/env python
"""N5: bootstrap CI for the +0.66 harm-vs-difficulty correlation + LaWAM-side recompute.

Original claim (PROGRESS_pi05_lmwm_P1_2026-07-23 §2.2):
  A1 (external DINOv3 space): corr(per-task Δ, difficulty) = +0.66 ("harder -> more harm")
  A2 (own So400m space):      corr = +0.06 (difficulty-independent)
  10 points = the 10 libero_10 tasks (t0..t9), pi05 4route x 50trial.

Metric reconstruction: Δ = arm_SR - baseline_SR per task; difficulty axis = baseline_SR
  (higher baseline = easier). corr(Δ, baseline) > 0  <=>  lower baseline (harder) -> more negative Δ
  = "harder -> more harm". We report Pearson & Spearman, bootstrap 95% CI (resample 10 tasks),
  and a permutation p-value. LaWAM side = same metric on Arm M (absolute milestone) vs Arm B
  over the SAME 10 libero_10 tasks (RESULTS_lmwm_vs_lawam_libero10_2026-07-15).
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
NB = 10000

# ---- pi05 P3 three-way (PROGRESS §1, %) ----
A0 = np.array([92.0, 98.7, 97.3, 94.0, 93.3, 98.7, 91.3, 98.0, 76.7, 98.7])  # baseline
A1 = np.array([94.0, 99.3, 98.7, 94.7, 97.3, 100., 89.3, 95.3, 66.7, 95.3])  # external DINOv3
A2 = np.array([92.5, 96.5, 99.0, 94.0, 95.5, 100., 92.0, 97.5, 75.0, 95.0])  # own So400m

# ---- LaWAM libero_10, Arm M (absolute milestone/dual) vs Arm B baseline (fractions) ----
M = np.array([0.98, 0.96, 1.00, 0.94, 1.00, 0.98, 0.68, 1.00, 0.86, 0.82]) * 100
B = np.array([0.98, 1.00, 1.00, 0.96, 0.98, 1.00, 0.84, 1.00, 1.00, 0.88]) * 100


def corr_ci(delta, difficulty, label):
    d = np.asarray(delta, float); x = np.asarray(difficulty, float)
    n = len(d)
    pear = stats.pearsonr(d, x); spear = stats.spearmanr(d, x)
    # bootstrap (resample task indices with replacement)
    bp, bs = [], []
    for _ in range(NB):
        idx = rng.integers(0, n, n)
        if np.std(d[idx]) < 1e-9 or np.std(x[idx]) < 1e-9:
            continue
        bp.append(np.corrcoef(d[idx], x[idx])[0, 1])
        bs.append(stats.spearmanr(d[idx], x[idx]).statistic)
    bp = np.array(bp); bs = np.array(bs)
    ci_p = (np.percentile(bp, 2.5), np.percentile(bp, 97.5))
    ci_s = (np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    frac_neg_p = float((bp < 0).mean())
    out = {
        "label": label, "n": n,
        "pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue),
        "pearson_boot_ci95": [float(ci_p[0]), float(ci_p[1])],
        "pearson_boot_frac_below_0": frac_neg_p,
        "spearman_r": float(spear.statistic), "spearman_p": float(spear.pvalue),
        "spearman_boot_ci95": [float(ci_s[0]), float(ci_s[1])],
        "spearman_boot_frac_below_0": float((bs < 0).mean()),
    }
    return out


def show(o):
    ci = o["pearson_boot_ci95"]; sci = o["spearman_boot_ci95"]
    cross = "CROSSES 0" if ci[0] <= 0 <= ci[1] else "excludes 0"
    scross = "CROSSES 0" if sci[0] <= 0 <= sci[1] else "excludes 0"
    print(f"\n=== {o['label']} (n={o['n']}) ===")
    print(f"  Pearson  r={o['pearson_r']:+.3f}  p={o['pearson_p']:.3f}  "
          f"boot95%=[{ci[0]:+.3f},{ci[1]:+.3f}] {cross}  P(r<0)={o['pearson_boot_frac_below_0']:.2f}")
    print(f"  Spearman r={o['spearman_r']:+.3f}  p={o['spearman_p']:.3f}  "
          f"boot95%=[{sci[0]:+.3f},{sci[1]:+.3f}] {scross}  P(r<0)={o['spearman_boot_frac_below_0']:.2f}")


results = []
print("difficulty axis = baseline SR (higher=easier); corr(Δ,baseline)>0 == harder->more harm")
for label, arm, base in [
    ("pi05 A1 external-DINOv3", A1, A0),
    ("pi05 A2 own-So400m",      A2, A0),
    ("LaWAM Arm M absolute-milestone", M, B),
]:
    o = corr_ci(arm - base, base, label)
    results.append(o); show(o)

import json
print("\n" + json.dumps(results, indent=2))

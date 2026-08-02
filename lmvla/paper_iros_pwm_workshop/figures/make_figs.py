#!/usr/bin/env python3
"""Workshop 论文三图 v2 — 空间分配原则:
注释只放确认为空的区域;图例移出绘图区;标题单行;每轴预留 headroom;
蓝#2a78d6=residual/ours, 橙#eb6834=absolute/harm, 灰=中性; 伤害/负值加 hatch。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})
BLUE, ORANGE, GRAY, DGRAY = "#2a78d6", "#eb6834", "#b7b6b0", "#52514e"
W = 3.5

# ================= Fig 1: plateau + per-task delta =================
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.0), width_ratios=[1, 1.12])
# ---- (a) 聚合平台。空区: 底部 92.5-93.6(点最低94.2-0.97≈93.2但x=0处93.2, 用x>1底部), 顶部右侧
names = ["base", "naive", "dual", "2Q", "tsched", "resid"]
agg   = [94.20, 94.6, 94.8, 94.80, 95.22, 95.75]
err   = [0.97, None, None, 0.71, 0.91, 0.76]
a.axhspan(94.20-1.5, 94.20+1.5, color=GRAY, alpha=0.22, lw=0, zorder=0)
a.axhline(94.20, color=DGRAY, lw=0.6, ls=":", zorder=1)
for i, (v, e) in enumerate(zip(agg, err)):
    c = BLUE if names[i] == "resid" else DGRAY
    if e: a.errorbar(i, v, yerr=e, fmt="o", color=c, mfc=c, ms=3.2, capsize=1.6, lw=0.9, zorder=3)
    else: a.plot(i, v, "o", color=c, mfc="white", ms=3.2, mew=0.9, zorder=3)
# 注释: claim band 标签放带内右下空区(x≥3.6,y≈93.0 无任何点/误差棒)
a.annotate("claim band\n(base ±1.5)", xy=(5.35, 92.95), fontsize=6, color=DGRAY,
           ha="right", va="bottom")
# ours 标签放最右点正上方 headroom(ylim 上界留 1.0)
a.annotate("ours", xy=(5, 95.75+0.76), xytext=(5, 97.15), fontsize=7, color=BLUE,
           ha="center", va="bottom",
           arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))
a.set_xticks(range(6), names, rotation=40, ha="right")
a.set_xlim(-0.6, 5.6); a.set_ylim(92.4, 97.9)
a.set_ylabel("LIBERO-10 aggregate SR (%)")
a.set_title("(a) mixings plateau", loc="left")
# ---- (b) per-task Δ。空区: 左下(t0-t3 为正, 其下方空), 顶部由 ylim 预留
dt = [1.3, 2.8, 1.5, 2.3, -2.2, 1.5, 5.0, 0.0, -2.5, 4.8]
cols = [BLUE if d > 0 else ORANGE for d in dt]
bars = b.bar(range(10), dt, color=cols, width=0.62, zorder=3)
for i, d in enumerate(dt):
    if d < 0: bars[i].set_hatch("///"); bars[i].set_edgecolor("white"); bars[i].set_linewidth(0)
b.axhline(0, color=DGRAY, lw=0.7)
for i in (6, 9):  # 指引任务标注: 柱顶上方 0.3, ylim 上界 7 留足
    b.annotate("guid.", xy=(i, dt[i]+0.3), ha="center", fontsize=6.2, color=BLUE)
# 方差说明放左下空区(t0-t2 下方), 无箭头避免穿柱
b.annotate("neg. bars: within\nbase var. (t8 ±12.4)", xy=(-0.35, -4.9), fontsize=5.8,
           color=DGRAY, ha="left", va="bottom")
b.set_xticks(range(10), [f"t{i}" for i in range(10)])
b.set_xlim(-0.7, 9.7); b.set_ylim(-5.2, 7.0)
b.set_ylabel(r"$\Delta$ SR (resid $-$ base, pts)", labelpad=1)
b.set_title("(b) guidance rises", loc="left")
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig("fig1_tension.pdf"); plt.close(fig)

# ================= Fig 2: redundancy =================
fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.8), width_ratios=[1.15, 1])
# ---- (a) CV 对比。图例移到轴上方外侧横排; 柱值标签上方留 headroom
eps = ["ep-moka", "ep-book"]
absv, resv = [0.119, 0.087], [1.112, 1.212]
x = np.arange(2); w = 0.30
r1 = a.bar(x-w/2, absv, w, color=ORANGE, hatch="///", edgecolor="white", lw=0, zorder=3, label="absolute")
r2 = a.bar(x+w/2, resv, w, color=BLUE, zorder=3, label="residual")
for r in (*r1, *r2):
    a.annotate(f"{r.get_height():.2f}", xy=(r.get_x()+r.get_width()/2, r.get_height()+0.045),
               ha="center", fontsize=6.2, color=DGRAY)
a.set_xticks(x, eps); a.set_xlim(-0.6, 1.6); a.set_ylim(0, 1.62)
a.set_ylabel("along-trajectory CV", labelpad=1)
a.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.04, 0.99),
         handlelength=1.0, columnspacing=0.9, borderaxespad=0.0, handletextpad=0.4)
a.set_title(r"(a) signal: ~10$\times$", loc="left", pad=16)
# ---- (b) 能量分解。文字全部放柱外: 上方灰说明, 下方蓝说明; 柱内不放字
b.barh([0], [86], color=GRAY, hatch="///", edgecolor="white", lw=0, height=0.42, zorder=3)
b.barh([0], [14], left=[86], color=BLUE, height=0.42, zorder=3)
b.annotate("86% = current state\n(redundant)", xy=(43, 0.34), ha="center", va="bottom",
           fontsize=6.2, color=DGRAY)
b.annotate("14% informative", xy=(86+7, -0.34), ha="center", va="top",
           fontsize=6.2, color=BLUE)
b.set_xlim(0, 102); b.set_ylim(-1.05, 1.05); b.set_yticks([])
b.set_xlabel("absolute-target energy (%)", labelpad=1)
b.set_title("(b) what it re-encodes", loc="left", pad=16)
fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig("fig2_redundancy.pdf"); plt.close(fig)

# ================= Fig 3: t5 anatomy =================
fig, ax = plt.subplots(figsize=(W, 1.6))
rows = ["no-WM baseline", "absolute targets", "absolute (indep. retrain)", "residual targets (ours)"]
vals = [76, 42, 18, 52]
cols = [DGRAY, ORANGE, ORANGE, BLUE]
y = np.arange(len(rows))[::-1]
bars = ax.barh(y, vals, color=cols, height=0.55, zorder=3)
bars[2].set_hatch("///"); bars[2].set_edgecolor("white"); bars[2].set_linewidth(0)
for yi, v in zip(y, vals):
    ax.annotate(f"{v}", xy=(v+1.5, yi), va="center", fontsize=7, color=DGRAY)
# 带区 94-100: 全列无柱(最大76), 竖排注释置于带中心、行方向居中
ax.axvspan(94, 100, color=GRAY, alpha=0.28, lw=0, zorder=0)
ax.annotate("other 9 tasks: 94–100 (all variants)", xy=(97, 1.5), ha="center", va="center",
            fontsize=5.8, color=DGRAY, rotation=90)
ax.set_yticks(y, rows)
ax.set_xlim(0, 106); ax.set_ylim(-0.55, 3.55)
ax.set_xlabel("t5 success rate (%)", labelpad=1)
fig.tight_layout(pad=0.4)
fig.savefig("fig3_t5.pdf"); plt.close(fig)
print("3 figs v2 written")

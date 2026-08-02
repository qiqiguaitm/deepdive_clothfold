#!/usr/bin/env python3
"""Generate the three main-paper figures from the current evidence table."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle

import palette

palette.apply_style()

BLUE = palette.NAT_BLUE
ORANGE = palette.NAT_ORANGE
GREEN = palette.NAT_TEAL
RED = palette.NAT_RED
GRAY = palette.GREY_300
DARK = palette.INK
LIGHT = palette.CANVAS


def box(ax, xy, width, height, text, face=LIGHT, edge=DARK, fontsize=6.8):
    patch = Rectangle(
        xy, width, height,
        linewidth=palette.LW_BASELINE,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
    )
    return patch


def arrow(ax, start, end, color=DARK, style="-", width=0.9, mutation=8):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=width,
        linestyle=style,
        color=color,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(patch)
    return patch


# Figure 1: actual MINT-VLA training and deployment pipeline.
fig, ax = plt.subplots(figsize=(5.45, 2.35))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.02, 0.94, "Offline target mining", fontsize=7, color=ORANGE, fontweight="bold")
box(ax, (0.02, 0.68), 0.17, 0.17, "successful\ndemonstrations", face="#FFF0EB", edge=ORANGE, fontsize=6.0)
box(ax, (0.23, 0.68), 0.18, 0.17, "frozen DINO\nrecurrence valleys", face="#FFF0EB", edge=ORANGE, fontsize=6.0)
box(ax, (0.45, 0.68), 0.17, 0.17, "store next\nmilestone frame", face="#FFF0EB", edge=ORANGE, fontsize=6.0)
arrow(ax, (0.19, 0.765), (0.23, 0.765), color=ORANGE)
arrow(ax, (0.41, 0.765), (0.45, 0.765), color=ORANGE)

ax.text(0.68, 0.94, "Training target", fontsize=7, color=BLUE, fontweight="bold")
box(ax, (0.68, 0.68), 0.13, 0.17, "current\nframe", face="#EAF3FA", edge=BLUE, fontsize=6.0)
box(ax, (0.85, 0.68), 0.13, 0.17, "stored\nframe", face="#EAF3FA", edge=BLUE, fontsize=6.0)
box(ax, (0.68, 0.43), 0.30, 0.15, "shared current visual encoder $E_\\theta$", face="#EAF3FA", edge=BLUE, fontsize=5.8)
arrow(ax, (0.745, 0.68), (0.78, 0.58), color=BLUE)
arrow(ax, (0.915, 0.68), (0.88, 0.58), color=BLUE)

box(ax, (0.36, 0.42), 0.23, 0.17, "$\\hat y=c+D_\\phi(c)$\nresidual-parameterized", face="#E8F5EE", edge=GREEN, fontsize=6.0)
arrow(ax, (0.68, 0.50), (0.59, 0.50), color=GREEN)
arrow(ax, (0.91, 0.43), (0.59, 0.43), color=ORANGE, style="--")
ax.text(0.74, 0.395, "$\\mathcal{L}_{ms}$; target stop-grad", fontsize=5.4, color=ORANGE)

box(ax, (0.08, 0.42), 0.20, 0.17, "one native VLM\nprefix token", face="#E8F5EE", edge=GREEN, fontsize=6.0)
arrow(ax, (0.36, 0.505), (0.28, 0.505), color=GREEN)
box(ax, (0.08, 0.17), 0.20, 0.14, "$\\pi_{0.5}$ action expert", face="#F3F3F0", edge=DARK, fontsize=6.1)
arrow(ax, (0.18, 0.42), (0.18, 0.31), color=DARK)

ax.text(0.36, 0.27, "Deployment", fontsize=7, color=GREEN, fontweight="bold")
box(ax, (0.36, 0.10), 0.62, 0.13,
    "observation $\\rightarrow E_\\theta \\rightarrow D_\\phi \\rightarrow$ one token; no retrieval, target encoder, or decoder",
    face="#E8F5EE", edge=GREEN, fontsize=5.7)
arrow(ax, (0.36, 0.165), (0.28, 0.235), color=GREEN)

fig.tight_layout(pad=0.25)
fig.savefig("fig1_tension.pdf")
plt.close(fig)


# Figure 2: primary pi0.5 task-level effects relative to matched A0.
tasks = ["Hammer", "Rank RGB", "Rank size", "Handover", "Stack-3", "Stack-2", "Macro"]
methods = ["A1 ext. abs.", "A2 offline abs.", "A2 offline resid.", "A3 live pred."]
deltas = np.array(
    [
        [9.5, 2.0, 0.5, -3.5, 4.5, 21.0, 5.67],
        [8.0, 14.5, 6.5, 10.5, 13.0, 27.0, 13.25],
        [6.5, 2.0, 10.5, 9.0, 7.5, 14.5, 8.33],
        [-4.0, 20.0, 1.5, 17.0, 18.0, 32.0, 14.08],
    ]
)
fig, ax = plt.subplots(figsize=(5.0, 1.9))
cmap = palette.DIVERGING_CMAP
norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=32)
im = ax.imshow(deltas, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(np.arange(len(tasks)), tasks, rotation=22, ha="right")
ax.set_yticks(np.arange(len(methods)), methods)
ax.axvline(5.5, color=DARK, lw=0.8)
for i in range(deltas.shape[0]):
    for j in range(deltas.shape[1]):
        value = deltas[i, j]
        color = "white" if value >= 18 or value <= -3.5 else DARK
        ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=6.2, color=color)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
cbar.set_label(r"$\Delta$ success vs. A0 (pts)", rotation=270, labelpad=10)
cbar.ax.tick_params(labelsize=6.1, width=0.5)
fig.tight_layout(pad=0.35)
fig.savefig("fig2_redundancy.pdf")
plt.close(fig)


# Figure 3: full task-level LaWAM diagnostic matrix relative to Future-off.
tasks = ["Hammer", "Rank RGB", "Rank size", "Handover", "Stack-3", "Stack-2"]
methods = ["Local future", "Absolute", "Residual", "Abs. + stop-grad", "Resid. + stop-grad"]
deltas = np.array(
    [
        [-4.5, -4.5, -8.5, -7.0, -2.5, 0.0],
        [-4.0, -6.0, -3.5, -3.5, 6.0, -0.5],
        [-6.5, -14.5, -3.0, -3.5, 9.0, -0.5],
        [-1.5, -7.0, -4.5, 1.0, 2.0, 0.0],
        [-4.0, -2.0, -6.0, -4.5, 5.5, -0.5],
    ]
)

fig, ax = plt.subplots(figsize=(4.8, 2.05))
cmap = palette.DIVERGING_CMAP
norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=9)
im = ax.imshow(deltas, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(np.arange(len(tasks)), tasks, rotation=24, ha="right")
ax.set_yticks(np.arange(len(methods)), methods)
ax.set_title("LaWAM task-level change relative to Future-off", loc="left", pad=4)
for i in range(deltas.shape[0]):
    for j in range(deltas.shape[1]):
        value = deltas[i, j]
        color = "white" if value <= -8 or value >= 7 else DARK
        label = "0.0" if value == 0 else f"{value:+.1f}"
        ax.text(j, i, label, ha="center", va="center", fontsize=6.4, color=color)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025)
cbar.set_label(r"$\Delta$ success (pts)", rotation=270, labelpad=9)
cbar.ax.tick_params(labelsize=6.2, width=0.5)
fig.tight_layout(pad=0.45)
fig.savefig("fig3_t5.pdf")
plt.close(fig)

print("Wrote fig1_tension.pdf, fig2_redundancy.pdf, and fig3_t5.pdf")

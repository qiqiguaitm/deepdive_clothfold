#!/usr/bin/env python3
"""Generate the claim-bearing main-paper figures from canonical evidence."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

import palette


palette.apply_style()

BLUE = palette.WONG_BLUE
ORANGE = palette.WONG_ORANGE
GREEN = palette.WONG_GREEN
RED = palette.WONG_VERMILLION
INK = palette.INK
GRAY = palette.GREY_500
LIGHT_GRAY = palette.GREY_200
WHITE = palette.CANVAS


def box(ax, xy, width, height, text, *, face=WHITE, edge=INK, fontsize=6.2, weight="regular"):
    patch = Rectangle(
        xy,
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=palette.LW_BASELINE,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
    )
    return patch


def arrow(ax, start, end, *, color=INK, linestyle="-", mutation=7):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=palette.LW_CONNECTOR,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


# ---------------------------------------------------------------------------
# Fig. 1: temporal contract, interface location, and evidence ladder.
# The saved width equals the ICLR text width used at insertion (5.45 in).
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(5.45, 3.05), facecolor=WHITE)

ax = fig.add_axes([0.055, 0.62, 0.90, 0.31])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
palette.panel_label(ax, -0.045, 1.00, "a")
ax.text(0.0, 1.00, "Temporal contract", fontsize=palette.FS_LABEL, fontweight="bold", va="bottom")

n_actions = 12
x0, x1 = 0.04, 0.73
cell_w = (x1 - x0) / n_actions
for i in range(n_actions):
    executed = i < 8
    ax.add_patch(
        Rectangle(
            (x0 + i * cell_w, 0.36),
            cell_w * 0.92,
            0.22,
            facecolor="#dbeaf4" if executed else "#eeeeee",
            edgecolor=BLUE if executed else GRAY,
            linewidth=palette.LW_BASELINE,
        )
    )
ax.text(x0, 0.67, "$t$", ha="center")
ax.text(x0 + 7.5 * cell_w, 0.67, "$t+E-1$", ha="center", color=BLUE)
ax.text(x0 + 11.5 * cell_w, 0.67, "$t+H-1$", ha="center", color=GRAY)
ax.plot([x0, x0 + 8 * cell_w], [0.24, 0.24], color=BLUE, lw=palette.LW_PRIMARY)
ax.plot([x0, x1], [0.13, 0.13], color=GRAY, lw=palette.LW_SECONDARY, ls="--")
ax.text(x0 + 4 * cell_w, 0.18, "$E$ actions executed", ha="center", color=BLUE, fontsize=6.2)
ax.text((x0 + x1) / 2, 0.01, "$H$ valid model actions", ha="center", color=GRAY, fontsize=6.2)
ax.plot([x0 + 7.5 * cell_w, x0 + 7.5 * cell_w], [0.58, 0.88], color=BLUE, lw=palette.LW_BASELINE)
ax.text(x0 + 7.5 * cell_w, 0.93, "execution endpoint", ha="center", color=BLUE, fontsize=6.0)
ax.plot([x0 + 11.5 * cell_w, x0 + 11.5 * cell_w], [0.58, 0.88], color=GRAY, lw=palette.LW_BASELINE)
ax.text(x0 + 11.5 * cell_w, 0.82, "model endpoint", ha="center", color=GRAY, fontsize=6.0)
arrow(ax, (0.77, 0.47), (0.94, 0.47), color=ORANGE)
ax.plot([0.94, 0.94], [0.35, 0.59], color=ORANGE, lw=palette.LW_PRIMARY)
ax.text(0.855, 0.67, "variable milestone", ha="center", color=ORANGE, fontsize=6.2)
ax.text(0.855, 0.25, r"$\tau(t)$ may be multi-query", ha="center", color=ORANGE, fontsize=6.0)

ax = fig.add_axes([0.055, 0.08, 0.58, 0.44])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
palette.panel_label(ax, -0.07, 1.00, "b")
ax.text(0.0, 1.00, "Conditioning location", fontsize=palette.FS_LABEL, fontweight="bold", va="bottom")

rows = [0.74, 0.45, 0.16]
names = ["LaWAM", "MINT-VLA", "Predictive\nadapter"]
for y, name in zip(rows, names):
    ax.text(0.0, y + 0.07, name, ha="left", va="center", fontsize=6.1, fontweight="bold", linespacing=0.95)

box(ax, (0.24, rows[0]), 0.18, 0.14, "current\ngrid", face="#dbeaf4", edge=BLUE)
box(ax, (0.50, rows[0]), 0.19, 0.14, "predicted endpoint\ngrid", face="#fce8d1", edge=ORANGE, fontsize=5.8)
box(ax, (0.78, rows[0]), 0.20, 0.14, "flow action\nexpert", face="#e1f1ea", edge=GREEN)
arrow(ax, (0.42, rows[0] + 0.07), (0.50, rows[0] + 0.07), color=ORANGE)
arrow(ax, (0.69, rows[0] + 0.07), (0.78, rows[0] + 0.07), color=GREEN)

box(ax, (0.24, rows[1]), 0.18, 0.14, "current\nfeature", face="#dbeaf4", edge=BLUE)
box(ax, (0.50, rows[1]), 0.19, 0.14, "one milestone\ntoken", face="#fce8d1", edge=ORANGE)
box(ax, (0.78, rows[1]), 0.20, 0.14, "VLM prefix +\naction expert", face="#eeeeee", edge=GRAY, fontsize=5.8)
arrow(ax, (0.42, rows[1] + 0.07), (0.50, rows[1] + 0.07), color=ORANGE)
arrow(ax, (0.69, rows[1] + 0.07), (0.78, rows[1] + 0.07), color=GRAY)

box(ax, (0.24, rows[2]), 0.18, 0.14, "current grid +\nnoisy proposal", face="#dbeaf4", edge=BLUE, fontsize=5.8)
box(ax, (0.50, rows[2]), 0.19, 0.14, "future-grid\nresidual", face="#fce8d1", edge=ORANGE)
box(ax, (0.78, rows[2]), 0.20, 0.14, "zero-init action\ntoken residual", face="#e1f1ea", edge=GREEN, fontsize=5.8)
arrow(ax, (0.42, rows[2] + 0.07), (0.50, rows[2] + 0.07), color=ORANGE)
arrow(ax, (0.69, rows[2] + 0.07), (0.78, rows[2] + 0.07), color=GREEN)

ax = fig.add_axes([0.69, 0.08, 0.265, 0.44])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
palette.panel_label(ax, -0.15, 1.00, "c")
ax.text(0.0, 1.00, "Evidence ladder", fontsize=palette.FS_LABEL, fontweight="bold", va="bottom")
ladder = [
    (0.77, "Predictability", BLUE),
    (0.53, "Matched utility", ORANGE),
    (0.29, "Content use", GREEN),
    (0.05, "Source attribution", RED),
]
for y, label, color in ladder:
    box(ax, (0.08, y), 0.84, 0.15, label, face=WHITE, edge=color, fontsize=6.1, weight="bold")
for upper, lower in zip(ladder[:-1], ladder[1:]):
    arrow(ax, (0.50, upper[0]), (0.50, lower[0] + 0.15), color=GRAY, linestyle="--")

fig.savefig("fig1_contract.pdf")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. 2: TG1A released-checkpoint content intervention.
# ---------------------------------------------------------------------------
conditions = ["Normal", "Shuffled", "Null", "Persistence"]
success = np.array([94.00, 40.33, 35.17, 24.42])
condition_colors = [BLUE, ORANGE, GRAY, palette.GREY_300]
hatches = ["", "//", "xx", ".."]

tasks = ["Hammer", "Rank RGB", "Rank size", "Handover", "Stack-2", "Stack-3"]
task_effects = np.array([17.0, 61.0, 67.0, 37.5, 64.5, 75.0])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(5.45, 2.35),
    gridspec_kw={"width_ratios": [0.93, 1.12]},
    constrained_layout=True,
)

ax = axes[0]
x = np.arange(len(conditions))
bars = ax.bar(
    x,
    success,
    width=0.66,
    color=condition_colors,
    edgecolor=INK,
    linewidth=palette.LW_BASELINE,
)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
for xi, value in zip(x, success):
    ax.text(xi, value + 2.0, f"{value:.2f}", ha="center", va="bottom", fontsize=6.1)
ax.set_xticks(x, conditions, rotation=24, ha="right")
ax.set_ylabel("Success (%)")
ax.set_ylim(0, 104)
ax.set_title("Fixed-checkpoint conditions", loc="left", pad=4)
ax.grid(axis="y")
palette.despine(ax)
palette.panel_label(ax, -0.19, 1.02, "a")

ax = axes[1]
y = np.arange(len(tasks))
ax.axvline(0, color=INK, linewidth=palette.LW_BASELINE)
for yi, effect in zip(y, task_effects):
    ax.plot([0, effect], [yi, yi], color=BLUE, linewidth=palette.LW_SECONDARY)
ax.scatter(
    task_effects,
    y,
    marker="o",
    s=20,
    facecolor=WHITE,
    edgecolor=BLUE,
    linewidth=palette.LW_SECONDARY,
    zorder=3,
)
for yi, effect in zip(y, task_effects):
    ax.text(effect + 1.8, yi, f"{effect:+.1f}", va="center", ha="left", fontsize=6.1)
ax.set_yticks(y, tasks)
ax.invert_yaxis()
ax.set_xlim(-2, 84)
ax.set_xlabel("Normal − shuffled success (pp)")
ax.set_title("All task effects", loc="left", pad=4)
ax.grid(axis="x")
palette.despine(ax)
palette.panel_label(ax, -0.22, 1.02, "b")

fig.savefig("fig2_content.pdf")
plt.close(fig)

print("Wrote fig1_contract.pdf and fig2_content.pdf")

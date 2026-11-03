#!/usr/bin/env python3
"""Article 4b, Fig. 1: abductive identity of a small target (replaces the pipeline scheme shared with 4a).

Left: frame neighbourhood with the prediction x̂_t, the admissible circle of radius L_t (one body length),
candidate b1 inside (the held target) and b2 outside (a neighbouring bird), classifier boxes on both.
Right: hypotheses same(b1), same(b2), unseen; the integrity constraint refutes same(b2); the minimal
admissible explanation keeps b1, and only b1's classifier label goes to the label accumulator and the
abductive class refinement.
Times New Roman, Ukrainian labels -> ../fig/fig1_identity_v7b.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Ellipse

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 14})


def bird(ax, x, y, s=1.0, c="0.15"):
    ax.add_patch(Ellipse((x, y), 0.9 * s, 0.45 * s, color=c, zorder=4))
    ax.plot([x - 0.45 * s, x - 1.3 * s], [y + 0.05 * s, y + 0.55 * s], color=c, lw=3.2 * s, solid_capstyle="round", zorder=4)
    ax.plot([x + 0.45 * s, x + 1.3 * s], [y + 0.05 * s, y + 0.55 * s], color=c, lw=3.2 * s, solid_capstyle="round", zorder=4)


def box(ax, x, y, w, h, text, fc, ec, fs=13, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", zorder=5)


def arrow(ax, x0, y0, x1, y1, c="0.15", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=c, linestyle=ls, shrinkA=0, shrinkB=0), zorder=2)


fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

# ---- left panel: frame neighbourhood
ax.add_patch(Rectangle((0.2, 0.5), 5.3, 5.3, fc="#dfe8f3", ec="0.4", lw=1.2, zorder=0))
ax.text(0.35, 5.55, "а", fontsize=16, weight="bold")
px, py = 2.4, 3.0
ax.add_patch(Circle((px, py), 1.25, fc="#c9e7d0", ec="#2e8b57", lw=2, ls="--", alpha=0.6, zorder=1))
ax.plot([px - 0.18, px + 0.18], [py, py], color="#d62728", lw=2.2, zorder=6)
ax.plot([px, px], [py - 0.18, py + 0.18], color="#d62728", lw=2.2, zorder=6)
ax.text(px - 0.25, py + 0.2, r"$\hat{x}_t$", fontsize=16, color="#d62728", ha="right")
ax.annotate("", xy=(px + 1.25, py), xytext=(px, py), arrowprops=dict(arrowstyle="<->", lw=1.2, color="#2e8b57"))
ax.text(px + 0.62, py - 0.38, r"$L_t$", fontsize=16, color="#2e8b57", ha="center")
bird(ax, 2.75, 3.65, 0.55)
ax.add_patch(Rectangle((2.15, 3.3), 1.2, 0.75, fill=False, ec="#e6b800", lw=2, zorder=5))
ax.text(2.75, 4.25, r"$b_1$", fontsize=16, ha="center")
bird(ax, 4.45, 1.45, 0.55)
ax.add_patch(Rectangle((3.85, 1.1), 1.2, 0.75, fill=False, ec="#e6b800", lw=2, zorder=5))
ax.text(4.45, 0.75, r"$b_2$ (сусідній птах)", fontsize=13, ha="center")
ax.text(2.85, 5.25, "окіл цілі в кадрі $t$", fontsize=13, ha="center")
ax.text(0.45, 0.62, "жовті рамки – мітки класифікатора", fontsize=11, color="0.3")

# ---- right panel: hypotheses and abductive choice
ax.text(5.95, 5.55, "б", fontsize=16, weight="bold")
ax.text(7.3, 5.55, "гіпотези тотожності", fontsize=13, ha="center")
box(ax, 7.3, 4.8, 2.4, 0.7, r"same($b_1$)", "#e8f5e9", "#2e8b57")
box(ax, 7.3, 3.7, 2.4, 0.7, r"same($b_2$)", "#fdecea", "#c0392b")
box(ax, 7.3, 2.6, 2.4, 0.7, "unseen (ціль невидима)", "#f3f3f3", "0.5", fs=12)
box(ax, 10.9, 4.25, 3.6, 1.05, "обмеження цілісності:\n" + r"$\|c_b-\hat{x}_t\| \leq L_t$", "white", "0.3", fs=13)
arrow(ax, 8.5, 4.8, 9.1, 4.45)
arrow(ax, 8.5, 3.7, 9.1, 4.05)
ax.text(10.9, 3.35, r"same($b_1$): допустима", fontsize=12, color="#2e8b57", ha="center")
ax.text(10.9, 2.95, r"same($b_2$): спростована ($b_2$ далі за $L_t$)", fontsize=12, color="#c0392b", ha="center")
box(ax, 10.9, 1.95, 3.6, 0.75, r"мінімальне пояснення: ціль $= b_1$", "#e8f5e9", "#2e8b57", fs=13)
arrow(ax, 10.9, 2.75, 10.9, 2.35, c="#2e8b57")
box(ax, 10.9, 0.75, 3.6, 0.95, r"мітка $b_1$ $\rightarrow$ мітки однієї цілі" + "\n" + r"$\rightarrow$ абдуктивне уточнення класу", "#eaf1fb", "#3a64a0", fs=12)
arrow(ax, 10.9, 1.57, 10.9, 1.24, c="#3a64a0")
ax.text(7.3, 1.75, "якщо всі same(b)\nспростовано –\nабдукується unseen:\nкадр без мітки", fontsize=11.5, ha="center", va="center", color="0.3")
fig.savefig(os.path.join(FIG, "fig1_identity_v7b.png"), dpi=300, bbox_inches="tight")
print("ok")

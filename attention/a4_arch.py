#!/usr/bin/env python3
"""Architecture of the classifier + attention coupling (a4_pipeline.py).
Black: links of the method as tested in drafts v1-v5; orange: links added by the proposed coupling.
Output: fig/fig_arch.png"""
import os
import sys

from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_style as S          # noqa: E402
from a4_style import plt      # noqa: E402
from common import A4         # noqa: E402

NEW = S.ORANGE


def blk(ax, x, y, w, h, t, ec="black", fc="white", fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=1.1 if ec != "black" else 0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arr(ax, pts, c="black", ls="-", t=None, tp=None, ha="center", rot=0):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=c, lw=1.1 if c != "black" else 0.9, ls=ls)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0,
                                                                 shrinkA=0, shrinkB=0, mutation_scale=11))
    if t:
        ax.text(*tp, t, fontsize=12, ha=ha, va="center", color=c, rotation=rot)


def main():
    W, H = S.W_IN, 7.4
    fig, ax = plt.subplots(figsize=(W, H)); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    blk(ax, 0.1, 6.55, 1.6, 0.7, "Кадр $I_t$")
    blk(ax, 2.3, 6.55, 2.2, 0.7, "Класифікатор,\nповний кадр")
    blk(ax, 4.9, 6.55, 1.5, 0.7, "Старт:\n$c\\geq 0{,}5$", ec=NEW)
    # window loop
    ax.add_patch(Rectangle((0.1, 3.1), 3.6, 3.05, fc=S.FILL_HOLD, ec="none"))
    ax.text(0.2, 5.95, "Вікно уваги (Holder)", fontsize=13, fontweight="bold", va="center")
    blk(ax, 0.3, 4.85, 1.45, 0.75, "Прогноз\n$\\hat{\\mathbf{x}}_t$")
    blk(ax, 2.05, 4.85, 1.45, 0.75, "Вікно $W_t$,\nпоріг $\\theta_W$")
    blk(ax, 2.05, 3.35, 1.45, 0.85, "Ворота $g$,\nінерція $K$")
    blk(ax, 0.3, 3.35, 1.45, 0.85, "Положення\n$\\mathbf{x}_t$")
    arr(ax, [(1.75, 5.22), (2.05, 5.22)]); arr(ax, [(2.77, 4.85), (2.77, 4.2)])
    arr(ax, [(2.05, 3.77), (1.75, 3.77)]); arr(ax, [(1.02, 4.2), (1.02, 4.85)])
    ax.text(0.95, 4.52, "$t+1$", fontsize=12, ha="right", va="center")
    # classifier on the window
    blk(ax, 4.1, 4.75, 2.3, 0.95, "Класифікатор\nна вирізці вікна\n$120\\to640$ пкс", ec=NEW)
    blk(ax, 4.1, 3.35, 2.3, 0.85, "Голоси класу\nвздовж треку", ec=NEW)
    # decision
    ax.add_patch(Rectangle((0.1, 0.15), 6.3, 2.65, fc=S.FILL_DEC, ec="none"))
    ax.text(0.2, 2.6, "Рішення", fontsize=13, fontweight="bold", va="center")
    blk(ax, 0.3, 1.25, 1.9, 0.85, "Ряд $w_t, h_t$\n$\\geq 48$ кадрів")
    blk(ax, 2.5, 1.25, 1.6, 0.85, "Критерій (6)\nмаховий політ")
    blk(ax, 4.4, 1.25, 1.9, 0.85, "Злиття:\nкритерій + голоси", ec=NEW)
    blk(ax, 2.1, 0.3, 4.2, 0.65, "«птах»  /  «БпЛА»  /  «невідомо»", ec=NEW)
    # links
    arr(ax, [(1.7, 6.9), (2.3, 6.9)])
    arr(ax, [(4.5, 6.9), (4.9, 6.9)], c=NEW)
    arr(ax, [(5.65, 6.55), (5.65, 6.3), (1.02, 6.3), (1.02, 5.6)], c=NEW, t="рамка цілі", tp=(3.3, 6.42))
    arr(ax, [(0.6, 6.55), (0.6, 5.6)], t="", tp=(0, 0))
    arr(ax, [(3.5, 5.22), (4.1, 5.22)], c=NEW, t="вирізка", tp=(3.8, 5.42))
    arr(ax, [(4.1, 4.95), (3.5, 4.95)], c="black", ls="--", t="підказка", tp=(3.8, 4.75))
    arr(ax, [(5.25, 4.75), (5.25, 4.2)], c=NEW)
    arr(ax, [(5.25, 3.35), (5.25, 2.1)], c=NEW)
    arr(ax, [(1.02, 3.35), (1.02, 2.1)])
    arr(ax, [(2.2, 1.67), (2.5, 1.67)])
    arr(ax, [(4.1, 1.67), (4.4, 1.67)], c=NEW)
    arr(ax, [(5.35, 1.25), (5.35, 0.95)], c=NEW)
    fig.savefig(os.path.join(A4, "fig", "fig_arch.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

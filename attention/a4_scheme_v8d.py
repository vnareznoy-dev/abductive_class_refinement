#!/usr/bin/env python3
"""Method scheme in the pipeline style (grouped coloured panels, symbols inside blocks).
v8d: no jargon ("ворота", "інерція"), theta_t as in formula (3). Output: fig/v8d/fig1_scheme.png"""
import os
import sys

from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_style as S          # noqa: E402
from a4_style import plt      # noqa: E402
from common import A4         # noqa: E402

H = 6.6


def blk(ax, x, y, w, h, text, fc="white", ec="black", lw=0.9, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arr(ax, pts, ls="-", text=None, tpos=None, ha="center"):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=0.9, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", shrinkA=0, shrinkB=0, mutation_scale=11))
    if text:
        ax.text(*tpos, text, fontsize=14, ha=ha, va="center")


def main():
    fig, ax = plt.subplots(figsize=(S.W_IN, H))
    ax.set_xlim(-0.3, S.W_IN); ax.set_ylim(0, H); ax.axis("off")
    blk(ax, 0.05, 5.55, 2.2, 0.85, "Кадр $I_t$, початкове\nположення цілі $\\mathbf{x}_0$")
    blk(ax, 4.15, 5.55, 2.3, 0.85, "Класифікатор або\nзовнішній датчик")
    # holding panel
    ax.add_patch(Rectangle((0.05, 1.25), 3.8, 4.0, fc=S.FILL_HOLD, ec="none"))
    ax.text(0.75, 5.0, "Утримання", fontsize=14, va="center", fontweight="bold")
    ax.text(0.75, 4.68, "(поріг вікна)", fontsize=14, va="center")
    blk(ax, 0.25, 3.6, 1.95, 0.8, "Прогноз\n$\\hat{\\mathbf{x}}_t=\\mathbf{x}_{t-1}+\\mathbf{v}_{t-1}$", fs=13)
    blk(ax, 0.25, 2.35, 1.95, 1.05, "Вікно $W_t$\n$120\\times120$ пкс,\n$\\theta_t=\\max(P_{99},\\,8)$", fs=13)
    blk(ax, 0.25, 1.4, 1.95, 0.75, "Є силует, де\n$\\|\\mathbf{c}_b-\\hat{\\mathbf{x}}_t\\|<g$ ?", fs=13)
    blk(ax, 2.45, 2.45, 1.25, 0.85, "Перезах-\nоплення", fs=13)
    blk(ax, 2.45, 1.4, 1.25, 0.75, "Екстрапо-\nляція,\n$m\\leq K$", fs=13)
    # decision panel
    ax.add_patch(Rectangle((4.05, 1.25), 2.4, 4.0, fc=S.FILL_DEC, ec="none"))
    ax.text(4.15, 5.0, "Абдуктивне рішення", fontsize=13, va="center", fontweight="bold")
    ax.text(4.15, 4.68, "(критерій [6], без змін)", fontsize=12, va="center")
    blk(ax, 4.2, 3.6, 2.1, 0.75, "Ряд $w_t,\\,h_t$,\n$\\geq 48$ кадрів", fs=13)
    blk(ax, 4.2, 2.6, 2.1, 0.65, "Спектр $P(f)$", fs=13)
    blk(ax, 4.2, 1.5, 2.1, 0.75, "$S>2S_w$,\n$A>0{,}02$ ?", fs=13)
    blk(ax, 0.25, 0.15, 1.95, 0.8, "Положення\nцілі $\\mathbf{x}_t$", fs=13)
    blk(ax, 4.2, 0.15, 2.1, 0.8, "«птах» або\n«невідомо»", fs=13)
    # main path
    arr(ax, [(0.5, 5.55), (0.5, 4.4)])
    arr(ax, [(1.22, 3.6), (1.22, 3.4)])
    arr(ax, [(1.22, 2.35), (1.22, 2.15)])
    arr(ax, [(1.22, 1.4), (1.22, 0.95)], text="так", tpos=(1.3, 1.12), ha="left")
    # loss branch
    arr(ax, [(2.2, 1.77), (2.45, 1.77)])
    ax.text(2.32, 1.7, "ні", fontsize=13, ha="center", va="top")
    arr(ax, [(3.07, 2.15), (3.07, 2.45)])
    ax.text(3.13, 2.3, "$m>K$", fontsize=13, ha="left", va="center")
    arr(ax, [(2.45, 2.87), (2.2, 2.87)])
    # cue
    arr(ax, [(4.15, 5.97), (3.07, 5.97), (3.07, 3.3)], ls="--")
    ax.text(3.14, 4.55, "підказка $(x,y)$", fontsize=13, ha="left", va="center", rotation=90)
    # loop to the next frame
    arr(ax, [(0.25, 0.55), (0.12, 0.55), (0.12, 4.0), (0.25, 4.0)])
    ax.text(0.05, 2.6, "$t+1$", fontsize=13, ha="right", va="center", rotation=90)
    # series to the decision
    arr(ax, [(2.2, 0.55), (3.95, 0.55), (3.95, 3.97), (4.2, 3.97)])
    ax.text(3.07, 0.68, "вихід методу уваги", fontsize=11, ha="center", va="bottom")
    arr(ax, [(5.25, 3.6), (5.25, 3.25)])
    arr(ax, [(5.25, 2.6), (5.25, 2.25)])
    arr(ax, [(5.25, 1.5), (5.25, 0.95)])
    fig.savefig(os.path.join(A4, "fig", "v8d", "fig1_scheme.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

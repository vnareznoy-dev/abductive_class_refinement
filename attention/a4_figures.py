#!/usr/bin/env python3
"""Figures of Article 4 (Ukrainian labels, Times New Roman 14 pt, 300 dpi).

fig/fig1_scheme.png   -- method scheme (two thresholds, holding ladder)
fig/fig2_conf.png     -- classifier confidence over time vs holding in the window
fig/fig3_frame.png    -- a real frame: detector silent, window holds the target
fig/fig4_modes.png    -- closed-loop holding strategies (E3)
"""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import A4, CACHE, REPORTS, MIDWAY, frames, local_blobs, load   # noqa: E402

FIG = os.path.join(A4, "fig")
os.makedirs(FIG, exist_ok=True)
TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
font_manager.fontManager.addfont(TNR)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14, "axes.linewidth": 0.8,
                     "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic"})
W_IN = 6.5          # 16.5 cm


# ------------------------------------------------------------------ fig 1
def box(ax, x, y, w, h, text, fill="white", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fill, ec="black", lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=14, fontweight="bold" if bold else "normal",
            linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, text=None, side="right"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", shrinkA=0, shrinkB=0))
    if text:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + (0.12 if side == "right" else -0.12), my, text, fontsize=14,
                ha="left" if side == "right" else "right", va="center")


def path(ax, pts, text=None, tpos=None):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=1.0)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black", shrinkA=0, shrinkB=0))
    if text:
        ax.text(*tpos, text, fontsize=14, ha="center", va="center")


def fig1():
    fig, ax = plt.subplots(figsize=(W_IN, 8.8))
    ax.set_xlim(0, 10.8); ax.set_ylim(0, 14); ax.axis("off")
    g = "#e6e6e6"
    box(ax, 5.2, 13.2, 6.4, 0.9, "Кадр відеопотоку, призначена ціль")
    box(ax, 2.4, 11.4, 4.2, 1.2, "Класифікатор або\nзовнішній датчик")
    box(ax, 7.9, 11.4, 4.4, 1.2, "Прогноз положення\n$\\hat{x}_t = x_{t-1} + v_{t-1}$")
    box(ax, 2.4, 9.2, 4.2, 1.5, "Підказка $(x, y)$:\nспалах $c_t \\geq \\theta_f$ або\nцілевказання")
    box(ax, 7.9, 9.2, 4.4, 1.5, "Вікно уваги $120 \\times 120$,\nпоріг утримання –\nлокальна статистика", fill=g)
    box(ax, 7.9, 6.9, 4.4, 1.2, "Силует у воротах:\n$\\| c_b - x_t \\| < g$ ?")
    box(ax, 7.9, 4.6, 4.4, 1.3, "Інерція: $m \\leq K$\nкадрів без силуету")
    box(ax, 2.4, 4.6, 4.2, 1.3, "Перезахоплення\nза підказкою")
    box(ax, 5.2, 2.5, 8.6, 1.1, "Неперервний ряд $w_t, h_t$ однієї цілі (48 кадрів)")
    box(ax, 5.2, 0.75, 8.6, 1.3, "Поріг рішення (незмінний):\nспектральний критерій, «птах» або «невідомо»", fill=g)
    path(ax, [(3.6, 12.75), (2.8, 12.0)])
    path(ax, [(6.8, 12.75), (7.6, 12.0)])
    path(ax, [(2.4, 10.8), (2.4, 9.95)])
    path(ax, [(7.9, 10.8), (7.9, 9.95)])
    path(ax, [(7.9, 8.45), (7.9, 7.5)])
    path(ax, [(7.9, 6.3), (7.9, 5.25)], "ні", (8.25, 5.78))
    path(ax, [(5.7, 4.3), (4.5, 4.3)], "$m > K$", (5.1, 3.9))
    path(ax, [(2.4, 8.45), (2.4, 5.25)])
    path(ax, [(4.5, 4.95), (5.0, 4.95), (5.0, 8.9), (5.7, 8.9)])
    path(ax, [(10.1, 6.9), (10.45, 6.9), (10.45, 2.5), (9.5, 2.5)], "так", (10.45, 7.35))
    path(ax, [(5.2, 1.95), (5.2, 1.4)])
    fig.savefig(os.path.join(FIG, "fig1_scheme.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ data for fig 2 / 3
def pick_examples():
    rows = list(csv.DictReader(open(os.path.join(REPORTS, "e1_e2.csv"))))
    out = []
    for lo, hi in ((4, 6), (7, 9), (10, 12)):
        rs = [r for r in rows if lo <= float(r["size_px"]) <= hi]
        med = np.median([float(r["cov_0.1"]) for r in rs])
        r = min(rs, key=lambda r: abs(float(r["cov_0.1"]) - med))
        out.append(int(r["track"]))
    return out


def series_for(tids):
    tracks = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {tid: {int(p[0]): p for p in tracks[tid]["pts"]} for tid in tids}
    res = {tid: [] for tid in tids}
    f1 = max(max(v) for v in truth.values())
    for k, fr in frames(MIDWAY, 0, f1):
        for tid in tids:
            if k in truth[tid]:
                p = truth[tid][k]
                near = [d for d in det.get(k, []) if np.hypot(d[0] - p[1], d[1] - p[2]) < 45]
                c = max((d[4] for d in near), default=0.0)
                v = any(np.hypot(b[0] - p[1], b[1] - p[2]) < 25 for b in local_blobs(fr, p[1], p[2]))
                res[tid].append((k, c, v, p))
    return res, tracks


def fig2(res, tids, tracks):
    fig, axes = plt.subplots(3, 1, figsize=(W_IN, 8.4), sharex=False)
    for ax, tid, lab in zip(axes, tids, "абв"):
        s = res[tid]
        f0 = s[0][0]
        x = np.array([k - f0 for k, *_ in s]); c = np.array([v[1] for v in s]); vis = [v[2] for v in s]
        sz = np.median([max(p[3], p[4]) for p in tracks[tid]["pts"]])
        in_h = False
        for xx, cc in zip(x, c):
            if cc < 0.1 and not in_h:
                in_h, h0 = True, xx
            if cc >= 0.1 and in_h:
                in_h = False; ax.axvspan(h0, xx, color="0.87", lw=0, zorder=0)
        if in_h:
            ax.axvspan(h0, x[-1], color="0.87", lw=0, zorder=0)
        ax.plot(x, c, color="black", lw=1.0)
        for th, ls in ((0.5, "--"), (0.25, "-."), (0.1, ":")):
            ax.axhline(th, color="black", lw=0.8, ls=ls)
        for xx, v in zip(x, vis):
            if v:
                ax.plot([xx, xx + 1], [-0.12, -0.12], color="black", lw=5, solid_capstyle="butt")
        ax.set_ylim(-0.2, 1.0); ax.set_xlim(0, x[-1] + 1)
        ax.set_yticks([0, 0.1, 0.25, 0.5, 1.0])
        ax.set_ylabel("$c_t$", labelpad=8)
        ax.set_title(f"{lab}", loc="left", fontsize=14)
        ax.text(0.99, 0.93, f"силует {sz:.0f} пкс", transform=ax.transAxes, ha="right", va="top", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Номер кадру від початку треку", labelpad=8)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig2_conf.png"), dpi=300)
    plt.close(fig)


def fig3(res, tid):
    s = res[tid]
    cand = [v for v in s if v[1] < 0.1 and v[2]]
    k, c, _, p = cand[len(cand) // 2]
    fr = None
    for kk, f in frames(MIDWAY, k, k):
        fr = f
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    tx, ty = int(p[1]), int(p[2])
    R = 150
    x0, y0 = max(0, tx - R), max(0, ty - R)
    crop = cv2.cvtColor(fr[y0:ty + R, x0:tx + R], cv2.COLOR_BGR2RGB)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W_IN, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    a.imshow(crop); a.set_xticks([]); a.set_yticks([])
    a.add_patch(plt.Rectangle((tx - x0 - 60, ty - y0 - 60), 120, 120, fill=False, ec="white", lw=1.5, ls="--"))
    a.add_patch(plt.Circle((tx - x0, ty - y0), 12, fill=False, ec="red", lw=1.5))
    for d in det.get(k, []):
        if abs(d[0] - tx) < R and abs(d[1] - ty) < R and d[4] >= 0.1:
            a.add_patch(plt.Rectangle((d[0] - d[2] / 2 - x0, d[1] - d[3] / 2 - y0), d[2], d[3],
                                      fill=False, ec="yellow", lw=1.2))
    a.set_title("а", loc="left", fontsize=14)
    z = cv2.resize(crop[ty - y0 - 14:ty - y0 + 14, tx - x0 - 14:tx - x0 + 14], (280, 280),
                   interpolation=cv2.INTER_NEAREST)
    b.imshow(z); b.set_xticks([]); b.set_yticks([])
    # what the attention window sees: pixels above the window-local threshold
    g = cv2.cvtColor(fr[ty - 60:ty + 60, tx - 60:tx + 60], cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = cv2.GaussianBlur(g, (0, 0), 7) - g
    thr = max(float(np.percentile(dark, 99.0)), 8.0)
    m = (dark > thr).astype(np.uint8)[60 - 14:60 + 14, 60 - 14:60 + 14]
    mz = cv2.resize(m, (280, 280), interpolation=cv2.INTER_NEAREST)
    b.contour(mz, levels=[0.5], colors="red", linewidths=1.5)
    b.set_title("б", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_frame.png"), dpi=300)
    plt.close(fig)
    return k, c


def fig4():
    rows = list(csv.DictReader(open(os.path.join(REPORTS, "e3.csv"))))
    modes = [("tbd", "СЗВ"), ("att_none", "У0"), ("att_contrast_self", "УК"),
             ("att_self", "УС"), ("att_ext0.2Hz", "УЗ-0,2"), ("att_ext1Hz", "УЗ-1")]
    cor, wr, no = [], [], []
    for m, _ in modes:
        rs = [r for r in rows if r["mode"] == m]
        n = len(rs)
        c = sum(r["correct"] == "True" for r in rs); d = sum(r["decided"] == "True" for r in rs)
        cor.append(100 * c / n); wr.append(100 * (d - c) / n); no.append(100 * (n - d) / n)
    fig, ax = plt.subplots(figsize=(W_IN, 4.2))
    x = np.arange(len(modes))
    ax.bar(x, cor, color="0.2", label="рішення щодо призначеної цілі")
    ax.bar(x, wr, bottom=cor, color="0.6", label="рішення щодо іншого об’єкта")
    ax.bar(x, no, bottom=np.array(cor) + np.array(wr), color="white", ec="black", lw=0.8, label="без рішення")
    for xi, v in zip(x, cor):
        if v > 0:
            ax.text(xi, v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in modes])
    ax.set_ylabel("Частка треків, %", labelpad=8); ax.set_ylim(0, 100)
    ax.set_xlabel("Стратегія утримання", labelpad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1, frameon=False, fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_modes.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return list(zip([l for _, l in modes], cor, wr, no))


if __name__ == "__main__":
    fig1()
    tids = pick_examples()
    res, tracks = series_for(tids)
    fig2(res, tids, tracks)
    k, c = fig3(res, tids[1])
    m = fig4()
    print("examples:", tids, "| fig3 frame", k, "conf", round(c, 3))
    for row in m:
        print(row)

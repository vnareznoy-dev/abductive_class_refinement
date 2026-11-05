#!/usr/bin/env python3
"""Figure «the UAV flies away»: working range of the classifier and of the attention window.

Top: one DUT drone (the segment with the median L0, its middle frame) reduced to L = 32 ... 4 px,
crop of 40x40 px of the analysed (reduced) frame, nearest-neighbour enlargement, the method mask
(red, formulas 2-3, window at the true position, component nearest to it) and the classifier
confidence c_t in that frame (formula 1). Bottom: E22 over the 24 segments versus L — classifier
coverage (median share of frames with c_t >= 0.25), visibility of the silhouette for the window
threshold (median share of frames, tolerance r = 2L) and strict held run on target (% of segments)
for СЗВ and У0. Output: fig/v8d/fig_range.png
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_drones as D            # noqa: E402
import a4_style as ST            # noqa: E402
from a4_style import plt         # noqa: E402
from a4_fig_sizes import crop, method_mask   # noqa: E402
from common import A4, CACHE, load            # noqa: E402

ST.apply()
SIZES = (32, 24, 16, 12, 9, 6, 4)
C = 20


def main():
    from ultralytics import YOLO
    model = YOLO(D.WEIGHTS)
    segs = D.segments(); L0s = []
    for i, idx in segs:
        fs, gt = D.load_seq(i); L0s.append(float(np.median([max(gt[j][2], gt[j][3]) for j in idx])))
    m = int(np.argsort(L0s)[len(L0s) // 2]); i, idx = segs[m]; L0 = L0s[m]
    fs, gt = D.load_seq(i); j = idx[len(idx) // 2]
    img = cv2.imread(fs[j]); x, y, w, h = gt[j]; tx, ty = x + w / 2, y + h / 2

    fig = plt.figure(figsize=(ST.W_IN, ST.W_IN * 0.78))
    gs = fig.add_gridspec(2, len(SIZES), height_ratios=[1, 2.1], hspace=0.35, wspace=0.08)
    for col, L in enumerate(SIZES):
        k = L0 / L
        W, H = int(round(img.shape[1] / k)), int(round(img.shape[0] / k))
        fr = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        r = model.predict(fr, conf=0.05, imgsz=max(64, int(math.ceil(max(W, H) / 32.0) * 32)), device="cpu", verbose=False)[0]
        c = max((float(b.conf[0]) for b in r.boxes
                 if np.hypot(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2) - tx / k, float((b.xyxy[0][1] + b.xyxy[0][3]) / 2) - ty / k) < D.GATE_DET),
                default=0.0)
        mk, wh, th = method_mask(fr, tx / k, ty / k, C)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(crop(fr, tx / k, ty / k, C), interpolation="nearest")
        ov = np.zeros((2 * C, 2 * C, 4)); ov[mk] = (1.0, 0.1, 0.1, 0.6); ax.imshow(ov, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{L} пкс", fontsize=9, pad=2)
        ax.text(0.5, -0.06, f"$c_t$={c:.2f}".replace(".", ","), transform=ax.transAxes, fontsize=8, ha="center", va="top")
        ax.text(0.5, -0.30, f"{wh[0]}×{wh[1]}", transform=ax.transAxes, fontsize=8, ha="center", va="top", color="#CC0000")
        print("strip L", L, "c", round(c, 2), "mask", wh)
    fig.axes[0].text(-0.18, 1.18, "а", transform=fig.axes[0].transAxes, fontsize=14, va="bottom")

    R = load(os.path.join(CACHE, "e22_range.pkl"))
    Ls = sorted({r["L"] for r in R}, reverse=True)
    med = lambda key: [100 * np.median([r[key] for r in R if r["L"] == L]) for L in Ls]
    pct = lambda key: [100 * np.mean([r[key] for r in R if r["L"] == L]) for L in Ls]
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(len(Ls))
    ax.plot(xs, med("cov"), "-o", color=ST.ORANGE, lw=1.4, ms=4, label="класифікатор бачить ціль ($c_t \\geq 0{,}25$), частка кадрів")
    ax.plot(xs, med("vis"), "-s", color=ST.GREEN, lw=1.4, ms=4, label="вікно уваги виділяє силует, частка кадрів")
    ax.plot(xs, pct("strict_att_none"), "-^", color=ST.BLUE, lw=1.4, ms=4, label="ряд на цілі: вікно уваги (У0), частка ділянок")
    ax.plot(xs, pct("strict_tbd"), "--v", color=ST.ORANGE, lw=1.2, ms=4, label="ряд на цілі: супровід за виявленнями, частка ділянок")
    ax.set_xticks(xs); ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("розмір цілі $L$, пкс (ціль віддаляється →)"); ax.set_ylabel("%"); ax.set_ylim(-3, 103)
    for n_, ys_ in enumerate((med("cov"), med("vis"), pct("strict_att_none"), pct("strict_tbd")), 1):   # numbers instead of a legend (journal rule)
        ax.text(xs[0] - 0.18, ys_[0] + {2: 3, 3: -3}.get(n_, 0), str(n_), fontsize=11, ha="right", va="center")
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.3)
    ST.panel(ax, "б")
    out = os.path.join(A4, "fig", "v8d"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_range.png"), dpi=300, bbox_inches="tight")
    for L, a_, b_, c_, d_ in zip(Ls, med("cov"), med("vis"), pct("strict_att_none"), pct("strict_tbd")):
        print(f"L={L}: cov {a_:.0f} vis {b_:.0f} У0 {c_:.0f} СЗВ {d_:.0f}")


if __name__ == "__main__":
    main()

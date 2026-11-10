#!/usr/bin/env python3
"""Fig. 4 (ІММ v8f+): how the attention window works on Anti-UAV, per background group.

For each group (E17: open / textured / notdark) ONE typical segment is chosen by rule: the
segment whose U0 on-target share (att_none, 9 px, tolerance 2L) is closest to the group median
(reports/e17_antiuav.csv). Its middle frame is processed exactly as in E17/a4_antiuav.run_segment:
HUD inpainting at full resolution, reduction to the equivalent size L = 9 px (factor k = L0/L),
then the attention window W_t of 120x120 px is placed at the TRUE position (visibility check).
Panels per row:
  1) window W_t of the reduced frame, circle of radius r = 2L around the true position;
  2) contrast D_t = G_sigma*I - I (formula 2), sigma = 7;
  3) mask D_t > theta_t, theta_t = max(P99(D_t | W_t), 8) (formula 3), connected regions;
     the region whose centre is within r of the true position is drawn green, others red;
     the panel states theta_t and whether the silhouette was found within r.
Output: fig/v8d/fig_attn_antiuav.png
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_antiuav as A                         # noqa: E402
import a4_style as ST                          # noqa: E402
from a4_style import plt                       # noqa: E402
from common import A4, CACHE, REPORTS, load   # noqa: E402

ST.apply()
L = 9
WIN = 60          # half-size of the 120x120 window
GROUPS = [("open", "однорідне\nтло"), ("textured", "неоднорідне\nтло"), ("notdark", "ціль не\nтемніша\nза тло")]


def typical(group):
    rows = [r for r in csv.DictReader(open(os.path.join(REPORTS, "e17_antiuav.csv")))
            if r["group"] == group and r["mode"] == "att_none" and int(r["size_px"]) == L]
    med = float(np.median([float(r["on2s"]) for r in rows]))
    r = min(rows, key=lambda r: (abs(float(r["on2s"]) - med), r["video"]))
    return os.path.join(A.AU, r["video"]), int(r["f0"]), med


def window(v, segs_idx):
    gt = A.gt_of(v); idx = segs_idx
    s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])); k = s0 / L
    j = idx[len(idx) // 2]
    cap = cv2.VideoCapture(v); cap.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = cap.read()
    f = A.clean(f, A.hud_mask(v))
    W, H = int(round(f.shape[1] / k)), int(round(f.shape[0] / k))
    fr = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    x, y, w, h = gt[j]; tx, ty = (x + w / 2) / k, (y + h / 2) / k
    # exactly as common.local_blobs: the window is CLIPPED by the reduced frame and the threshold
    # is the percentile over the pixels inside the frame only; outside the frame is shown white.
    x0, x1 = max(0, int(tx - WIN)), min(W, int(tx + WIN)); y0, y1 = max(0, int(ty - WIN)), min(H, int(ty + WIN))
    sub = fr[y0:y1, x0:x1]
    g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Dsub = cv2.GaussianBlur(g, (0, 0), 7) - g
    th = max(float(np.percentile(Dsub, 99)), 8.0)
    m = (Dsub > th).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    ox, oy = int(tx - WIN), int(ty - WIN)                     # window origin in frame coordinates
    win = np.full((2 * WIN, 2 * WIN, 3), 255, np.uint8); D = np.zeros((2 * WIN, 2 * WIN), np.float32)
    col = np.full((2 * WIN, 2 * WIN, 3), 255, np.uint8)
    sl = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
    win[sl] = sub; D[sl] = np.clip(Dsub, 0, None)
    cm = np.full(sub.shape, 235, np.uint8)                     # frame area in light grey
    tw = (tx - ox, ty - oy); found = False
    for i in range(1, n):
        near = np.hypot(cen[i][0] + x0 - tx, cen[i][1] + y0 - ty) < 2 * L
        found |= near
        cm[lab == i] = (0, 158, 115) if near else (213, 94, 0)
    col[sl] = cm
    return cv2.cvtColor(win, cv2.COLOR_BGR2RGB), D, col, th, tw, found, k


def main():
    segs = load(os.path.join(CACHE, "e17_segments.pkl"))
    fig, axes = plt.subplots(3, 3, figsize=(ST.W_IN, ST.W_IN * 1.02))
    titles = ["вікно уваги $W_t$", "контраст $D_t$", "маска $D_t>\\theta_t$"]
    for r, (g, lab) in enumerate(GROUPS):
        v, f0, med = typical(g)
        idx = next(i for gg, vv, i in segs if gg == g and vv == v and i[0] == f0)
        win, D, col, th, tw, found, k = window(v, idx)
        circ = lambda ax: ax.add_patch(plt.Circle(tw, 2 * L, fill=False, ec="#CC0000", lw=1.2))
        ax = axes[r, 0]; ax.imshow(win, interpolation="nearest"); circ(ax)
        ax.set_ylabel(lab, rotation=0, labelpad=30, va="center", ha="center", fontsize=10)
        ax = axes[r, 1]; ax.imshow(D, cmap="gray_r", interpolation="nearest"); circ(ax)
        ax = axes[r, 2]; ax.imshow(col, interpolation="nearest"); circ(ax)
        txt = f"$\\theta_t$ = {th:.0f}".replace(".", ",") + ("\nсилует виділено" if found else "\nсилует не виділено")
        ax.text(3, 6, txt, fontsize=8, va="top", ha="left", bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        print(g, os.path.relpath(v, A.AU), "f0", f0, "k %.1f" % k, "theta %.1f" % th, "found", found, "group median on2s %.2f" % med)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.4)
    out = os.path.join(A4, "fig", "v8d"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_attn_antiuav.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

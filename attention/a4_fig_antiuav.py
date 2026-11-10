#!/usr/bin/env python3
"""v8e: group names on the figure, red circle = target, 4 tiles per group.
Figure of Anti-UAV examples per background group (E17), chosen by rule, not by hand:
for each group the segments are taken in the order of cache/e17_segments.pkl and 6 are picked
evenly (indices round(linspace(0, n-1, 6))); the tile is the middle frame of the segment,
crop of 8 target sizes around the true position, HUD inpainted as in E17.
Output: fig/v8d/fig_antiuav.png"""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_antiuav as A                         # noqa: E402
import a4_style as ST                          # noqa: E402
from a4_style import plt                       # noqa: E402
from common import A4, CACHE, load             # noqa: E402

ST.apply()
segs = load(os.path.join(CACHE, "e17_segments.pkl"))
groups = [("open", "однорідне\nтло"), ("textured", "неоднорідне\nтло"), ("notdark", "ціль не\nтемніша\nза тло")]
NCOL = 4
fig, axes = plt.subplots(3, NCOL, figsize=(ST.W_IN, ST.W_IN * 0.78))
for r, (g, lab) in enumerate(groups):
    gs = [s for s in segs if s[0] == g]
    pick = [gs[i] for i in np.round(np.linspace(0, len(gs) - 1, NCOL)).astype(int)]
    for c, (_, v, idx) in enumerate(pick):
        gt = A.gt_of(v); j = idx[len(idx) // 2]
        cap = cv2.VideoCapture(v)
        for _ in range(j + 1):          # sequential decoding: seeking in mp4 lands off-frame
            ok, f = cap.read()
        f = A.clean(f, A.hud_mask(v))
        x, y, w, h = gt[j]; cx, cy, rr = int(x + w / 2), int(y + h / 2), int(max(w, h) * 4)
        f = cv2.copyMakeBorder(f, rr, rr, rr, rr, cv2.BORDER_CONSTANT, value=(255, 255, 255))   # target near the edge: keep it centred
        cx, cy = cx + rr, cy + rr; y0, x0 = cy - rr, cx - rr
        crop = cv2.cvtColor(f[y0:cy + rr, x0:cx + rr], cv2.COLOR_BGR2RGB).copy()
        crop = cv2.resize(crop, (400, 400)); sc = 400.0 / (2 * rr)
        cv2.circle(crop, (int((cx - x0) * sc), int((cy - y0) * sc)), int(max(w, h) * sc * 0.9), (220, 30, 30), 3)
        ax = axes[r, c]; ax.imshow(crop); ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(lab, rotation=0, labelpad=34, va="center", ha="center", fontsize=11)
fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
out = os.path.join(A4, "fig", "v8d"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_antiuav.png"), dpi=300, bbox_inches="tight"); print("ok")

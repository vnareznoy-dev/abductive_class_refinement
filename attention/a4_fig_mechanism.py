#!/usr/bin/env python3
"""Prototype figure «how the window threshold works» (profile + histogram), ІММ v8g+.

Frame: the frame of Fig. 3 (a4_figures.fig3 rule: track tids[1] of pick_examples, the middle
frame among those where the classifier is silent (c < 0.1) and the silhouette is visible).
Nothing in the image is changed; D is computed exactly as in the method and in the baseline:
  window : D = G_7 * I - I over the 120x120 window centred on the target, theta_t = max(P99, 8)
           (common.local_blobs, formula (3));
  frame  : D = G_9 * I - I over the full frame with subtitle/top bands masked, threshold
           max(P99.9, 8) (flap_real.dark_blobs, the full-frame baseline of the article).
Panels: а) frame fragment, window, profile line; б) brightness I and background G*I along the line;
        в) D along the line with theta_t, theta_frame and theta_0; г) histograms of D in the window
        and in the frame with both thresholds and the target value.
Output: fig/v8d/fig_mechanism.png (+ printed numbers)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_figures as F           # noqa: E402
import a4_style as ST            # noqa: E402
from a4_style import plt         # noqa: E402
from common import A4, MIDWAY, frames   # noqa: E402

ST.apply()
WIN = 60


def main():
    tids = F.pick_examples()
    res, tracks = F.series_for([tids[1]])
    s = res[tids[1]]
    cand = [v for v in s if v[1] < 0.1 and v[2]]
    k, c, _, p = cand[len(cand) // 2]
    fr = None
    for _, f in frames(MIDWAY, k, k):
        fr = f
    tx, ty = int(round(p[1])), int(round(p[2]))
    G = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # window, as in the method
    g = G[ty - WIN:ty + WIN, tx - WIN:tx + WIN]
    bg = cv2.GaussianBlur(g, (0, 0), 7); Dw = bg - g
    th_w = max(float(np.percentile(Dw, 99)), 8.0)
    # full frame, as in the baseline
    Gf = G.copy(); H, W = Gf.shape
    Gf[int(0.82 * H):, :] = 255; Gf[:int(0.05 * H), :] = 255
    Df = cv2.GaussianBlur(Gf, (0, 0), 9) - Gf
    th_f = max(float(np.percentile(Df, 99.9)), 8.0)
    # profile through the darkest pixel of the silhouette (within 6 px of the true position)
    sub = g[WIN - 6:WIN + 7, WIN - 6:WIN + 7]
    dy, dx = np.unravel_index(np.argmin(sub), sub.shape); row = WIN - 6 + dy; col0 = WIN - 6 + dx
    xs = np.arange(-40, 41)
    I_line = g[row, col0 + xs]; B_line = bg[row, col0 + xs]; D_line = Dw[row, col0 + xs]
    Df_line = Df[ty - WIN + row, tx - WIN + col0 + xs]
    d_target = float(Dw[row, col0]); df_target = float(Df[ty - WIN + row, tx - WIN + col0])
    print(f"frame {k}, track {tids[1]}, classifier c = {c:.3f}")
    print(f"window: theta_t = {th_w:.1f}; frame: theta = {th_f:.1f}; target D window {d_target:.1f}, frame {df_target:.1f}")

    fig = plt.figure(figsize=(ST.W_IN, ST.W_IN * 0.78))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], hspace=0.5, wspace=0.38)
    # а) fragment
    a = fig.add_subplot(gs[0, 0]); R = 90
    crop = cv2.cvtColor(fr[ty - R:ty + R, tx - R:tx + R], cv2.COLOR_BGR2RGB)
    a.imshow(crop); a.set_xticks([]); a.set_yticks([])
    a.add_patch(plt.Rectangle((R - WIN, R - WIN), 2 * WIN, 2 * WIN, fill=False, ec="white", lw=1.2, ls="--"))
    from common import CACHE, load
    for d in load(os.path.join(CACHE, "det_midway.pkl")).get(k, []):   # classifier boxes (conf >= 0.1)
        if d[4] >= 0.1 and abs(d[0] - tx) < R and abs(d[1] - ty) < R:
            a.add_patch(plt.Rectangle((d[0] - d[2] / 2 - tx + R, d[1] - d[3] / 2 - ty + R), d[2], d[3], fill=False, ec="#E6C200", lw=1.2))
    yy = R - WIN + row
    a.plot([R - WIN + col0 - 40, R - WIN + col0 + 40], [yy, yy], color=ST.ORANGE, lw=1.2)
    a.set_xlim(0, 2 * R); a.set_ylim(2 * R, 0)
    ST.panel(a, "а")
    # б) brightness profile
    b = fig.add_subplot(gs[0, 1])
    b.plot(xs, I_line, color="black", lw=1.2, label="яскравість $I$")
    b.plot(xs, B_line, color=ST.BLUE, lw=1.2, ls="--", label="фон $G_\\sigma * I$")
    b.set_xlabel("відстань від цілі, пкс"); b.set_ylabel("рівень яскравості")
    b.legend(loc="lower left", fontsize=8, frameon=True); ST.panel(b, "б"); ST.comma(b)
    # в) D profile with thresholds
    v = fig.add_subplot(gs[1, 0])
    v.plot(xs, D_line, color="black", lw=1.2, label="$D=G_\\sigma * I-I$")
    v.axhline(th_w, color=ST.GREEN, lw=1.6, label=f"поріг вікна $\\theta_t$ = {th_w:.0f}")
    v.axhline(th_f, color=ST.ORANGE, lw=1.2, ls="--", label=f"поріг кадру = {th_f:.0f}")
    v.set_xlabel("відстань від цілі, пкс"); v.set_ylabel("$D$, рівнів")
    v.set_ylim(-15, 165); v.legend(loc="upper left", fontsize=7.5, frameon=True, ncol=1); ST.panel(v, "в"); ST.comma(v)
    # г) histograms
    h = fig.add_subplot(gs[1, 1])
    bins = np.arange(-20, max(th_f, d_target) + 12, 2)
    h.hist(Dw.ravel(), bins=bins, density=True, histtype="step", color=ST.GREEN, lw=1.2, label="вікно 120×120")
    h.hist(Df.ravel(), bins=bins, density=True, histtype="step", color=ST.ORANGE, lw=1.2, label="увесь кадр")
    h.set_yscale("log")
    h.axvline(th_w, color=ST.GREEN, lw=1.6); h.axvline(th_f, color=ST.ORANGE, lw=1.0, ls="--")
    h.axvline(d_target, color="black", lw=1.0, ls=":")
    h.annotate("ціль", xy=(d_target, 0.02), xycoords=("data", "axes fraction"), xytext=(d_target - 4, 0.62), textcoords=("data", "axes fraction"), fontsize=8, ha="right", arrowprops=dict(arrowstyle="->", lw=0.8))
    h.set_xlabel("$D$, рівнів"); h.set_ylabel("частка пікселів")
    h.legend(loc="upper right", fontsize=7.5, frameon=True); ST.panel(h, "г")
    out = os.path.join(A4, "fig", "v8d"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_mechanism.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

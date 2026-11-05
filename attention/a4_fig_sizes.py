#!/usr/bin/env python3
"""Figure «what the method sees on a target of a few pixels» for Tables 1–2 (ІММ v8i+).

Every tile: an enlarged crop with a pixel grid (one cell = one pixel of the analysed frame) and,
in translucent red, the pixels the method selects: D = G_7*I - I > theta_t, theta_t = max(P99, 8)
over the 120x120 attention window placed at the true position (formulas 2-3), closing 3x3, the
connected component nearest to the true position. Its width x height in pixels are exactly the
values w_t, h_t of the size series used by the decision criterion (6).
Row 1 (birds, native 1920x1080 Midway frames): the typical track of each size bin of Table 1
(a4_figures.pick_examples), middle frame; the crop is large enough to show the whole bird, so the
reader sees which part of it the method takes (only parts darker than the sky).
Row 2 (UAV, DUT): the segment whose L0 is the median over the 24 segments, middle frame; the
original and the frame reduced to L = 12 / 9 / 6 px (INTER_AREA); dashed = annotation box (size L).
c_t: classifier confidence in that frame (formula 1).
Output: fig/v8d/fig_sizes.png
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_drones as D            # noqa: E402
import a4_figures as F           # noqa: E402
import a4_style as ST            # noqa: E402
from a4_style import plt         # noqa: E402
from common import A4, CACHE, MIDWAY, frames, load   # noqa: E402

ST.apply()
CB, CU = 36, 16   # half crop for birds / reduced UAV frames, px of the analysed frame


def pad(img, h):
    return cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REPLICATE)


def method_mask(img, x, y, c):
    """Mask of the method around (x, y), returned in the coordinates of a (2c x 2c) crop."""
    P = 60 + c
    g = cv2.cvtColor(pad(img, P), cv2.COLOR_BGR2GRAY).astype(np.float32)
    xi, yi = int(round(x)) + P, int(round(y)) + P
    w = g[yi - 60:yi + 60, xi - 60:xi + 60]
    Dw = cv2.GaussianBlur(w, (0, 0), 7) - w
    th = max(float(np.percentile(Dw, 99)), 8.0)
    m = (Dw > th).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros((2 * c, 2 * c), bool), (0, 0), th
    i = 1 + int(np.argmin([np.hypot(q[0] - 60, q[1] - 60) for q in cen[1:]]))
    big = np.zeros((120 + 2 * c, 120 + 2 * c), bool)
    big[c:c + 120, c:c + 120] = lab == i
    return big[60:60 + 2 * c, 60:60 + 2 * c], (int(st[i][2]), int(st[i][3])), th


def crop(img, x, y, c):
    g = pad(img, c); x, y = int(round(x)) + c, int(round(y)) + c
    return cv2.cvtColor(g[y - c:y + c, x - c:x + c], cv2.COLOR_BGR2RGB)


def conf_at(dets, x, y):
    return max((d[4] for d in dets if np.hypot(d[0] - x, d[1] - y) < 45), default=0.0)


def tile(ax, im, mask, wh, title, c_t, box=None):
    n = im.shape[0]
    ax.imshow(im, interpolation="nearest")
    ov = np.zeros((n, n, 4)); ov[mask] = (1.0, 0.1, 0.1, 0.55)
    ax.imshow(ov, interpolation="nearest")
    for v in range(n + 1):
        ax.axhline(v - 0.5, color="white", lw=0.2, alpha=0.3); ax.axvline(v - 0.5, color="white", lw=0.2, alpha=0.3)
    if box is not None:
        bw, bh = box
        ax.add_patch(plt.Rectangle((n / 2 - bw / 2 - 0.5, n / 2 - bh / 2 - 0.5), bw, bh, fill=False, ec="#FFD000", lw=1.1, ls="--"))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10, pad=3)
    ax.text(1, 1, f"маска {wh[0]}×{wh[1]} пкс", color="#FF5050", fontsize=8.5, ha="left", va="top",
            bbox=dict(fc="black", ec="none", alpha=0.6, pad=0.8))
    ax.text(n - 2, n - 2, f"$c_t$ = {c_t:.2f}".replace(".", ","), color="white", fontsize=8.5, ha="right", va="bottom",
            bbox=dict(fc="black", ec="none", alpha=0.6, pad=0.8))


def main():
    fig, axes = plt.subplots(2, 4, figsize=(ST.W_IN, ST.W_IN * 0.58))
    # ---- birds
    tids = F.pick_examples()
    tracks = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    want = {}
    for tid in tids:
        p = tracks[tid]["pts"][len(tracks[tid]["pts"]) // 2]; want[int(p[0])] = (tid, p)
    got = {}
    for k, fr in frames(MIDWAY, min(want), max(want)):
        if k in want:
            got[k] = fr.copy()
    order = sorted(want, key=lambda k: np.median([max(q[3], q[4]) for q in tracks[want[k][0]]["pts"]]))
    for col, k in enumerate(order):
        tid, p = want[k]
        mk, wh, th = method_mask(got[k], p[1], p[2], CB)
        tile(axes[0, col], crop(got[k], p[1], p[2], CB), mk, wh, f"птах {col + 1}", conf_at(det.get(k, []), p[1], p[2]))
        print("bird", tid, "frame", k, "mask w x h", wh, "theta", round(th, 1))
    k = order[1]; tid, p = want[k]
    ctx = crop(got[k], p[1], p[2], 200)
    ax = axes[0, 3]; ax.imshow(ctx); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((200 - CB, 200 - CB), 2 * CB, 2 * CB, fill=False, ec="red", lw=1.0))
    ax.set_title("птах 2 у кадрі, 400×400", fontsize=10, pad=3)
    # ---- UAV
    segs = D.segments(); L0s = []
    for i, idx in segs:
        fs, gt = D.load_seq(i); L0s.append(float(np.median([max(gt[j][2], gt[j][3]) for j in idx])))
    m = int(np.argsort(L0s)[len(L0s) // 2]); i, idx = segs[m]; L0 = L0s[m]
    fs, gt = D.load_seq(i); j = idx[len(idx) // 2]
    img = cv2.imread(fs[j]); x, y, w, h = gt[j]; tx, ty = x + w / 2, y + h / 2
    dU = load(os.path.join(CACHE, "e6_det_uav_ours.pkl"))
    ax = axes[1, 0]; r0 = int(L0 * 1.5); ax.imshow(crop(img, tx, ty, r0)); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Rectangle((r0 - w / 2, r0 - h / 2), w, h, fill=False, ec="#FFD000", lw=1.1, ls="--"))
    ax.set_title(f"БпЛА, оригінал {L0:.0f} пкс", fontsize=10, pad=3)
    for col, L in zip((1, 2, 3), (12, 9, 6)):
        kk = L0 / L
        fr = cv2.resize(img, (int(round(img.shape[1] / kk)), int(round(img.shape[0] / kk))), interpolation=cv2.INTER_AREA)
        mk, wh, th = method_mask(fr, tx / kk, ty / kk, CU)
        c = conf_at(dU.get((i, idx[0], L, j), []), tx / kk, ty / kk)
        tile(axes[1, col], crop(fr, tx / kk, ty / kk, CU), mk, wh, f"зменшено до L = {L} пкс", c, box=(w / kk, h / kk))
        print("uav L", L, "mask w x h", wh, "theta", round(th, 1), "c", round(c, 2))
    print("UAV segment", i, idx[0], "L0", round(L0, 1), "frame", j, "| bird tracks", tids)
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.8)
    out = os.path.join(A4, "fig", "v8d"); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_sizes.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gallery of small targets: birds (Midway, real distance) and UAVs (DUT, equivalent
range by frame reduction), eight frames with step 2, nearest-neighbour upscaling.
Output: fig/fig_gallery.png"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import A4, CACHE, MIDWAY, frames, load   # noqa: E402
import a4_drones as D                                 # noqa: E402

TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
font_manager.fontManager.addfont(TNR)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})
R, NF, STEP = 14, 8, 2


def bird_rows():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    sz = {t["id"]: np.median([max(p[3], p[4]) for p in t["pts"]]) for t in tracks}
    # tracks without neighbouring birds in the crop (checked on reports/sheets)
    by_id = {t["id"]: t for t in tracks}
    pick = [(int(round(sz[i])), by_id[i]) for i in (44, 57, 12, 53)]
    want = {}
    for target, t in pick:
        mid = len(t["pts"]) // 3
        for j in range(NF):
            p = t["pts"][mid + j * STEP]
            want.setdefault(int(p[0]), []).append((target, j, p))
    tiles = {target: [None] * NF for target, _ in pick}
    for k, fr in frames(MIDWAY, min(want), max(want)):
        for target, j, p in want.get(k, []):
            cx, cy = int(p[1]), int(p[2])
            tiles[target][j] = cv2.cvtColor(fr[cy - R:cy + R, cx - R:cx + R], cv2.COLOR_BGR2RGB)
    return [(f"{target} пкс", tiles[target]) for target, _ in pick]


def uav_rows():
    segs = D.segments()
    i, idx = max(segs, key=lambda s: len(s[1]))
    fs, gt = D.load_seq(i)
    s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx]))
    rows = []
    for S in (6, 9, 12):
        k = s0 / S
        tl = []
        mid = len(idx) // 3
        for jj in range(NF):
            j = idx[mid + jj * STEP]
            im = cv2.imread(fs[j])
            sm = cv2.resize(im, (int(round(im.shape[1] / k)), int(round(im.shape[0] / k))), interpolation=cv2.INTER_AREA)
            cx, cy = int((gt[j][0] + gt[j][2] / 2) / k), int((gt[j][1] + gt[j][3] / 2) / k)
            crop = np.full((2 * R, 2 * R, 3), 255, np.uint8)
            y0, x0 = max(0, cy - R), max(0, cx - R)
            part = sm[y0:cy + R, x0:cx + R]
            crop[(y0 - (cy - R)):(y0 - (cy - R)) + part.shape[0], (x0 - (cx - R)):(x0 - (cx - R)) + part.shape[1]] = part
            tl.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        rows.append((f"{S} пкс", tl))
    return rows, i


def main():
    b = bird_rows()
    u, seq = uav_rows()
    allrows = [("а", r) for r in b] + [("б", r) for r in u]
    fig, axes = plt.subplots(len(allrows), NF, figsize=(6.5, 0.86 * len(allrows) + 0.3))
    for ri, (panel, (lab, tl)) in enumerate(allrows):
        for j in range(NF):
            ax = axes[ri, j]
            ax.imshow(cv2.resize(tl[j], (8 * 2 * R, 8 * 2 * R), interpolation=cv2.INTER_NEAREST))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
        axes[ri, 0].set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=6, fontsize=14)
    axes[0, 0].set_title("а", loc="left", fontsize=14, x=-0.9)
    axes[len(b), 0].set_title("б", loc="left", fontsize=14, x=-0.9)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.96, bottom=0.01, wspace=0.06, hspace=0.12)
    fig.savefig(os.path.join(A4, "fig", "fig_gallery.png"), dpi=300)
    print("saved; UAV sequence", seq)


if __name__ == "__main__":
    main()

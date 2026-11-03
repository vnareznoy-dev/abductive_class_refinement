#!/usr/bin/env python3
"""Diagnostic of the false 'bird' decisions on UAVs (E5).

For every DUT open-sky segment at S = 9 px, the silhouette series is taken
at the TRUE position (nearest window blob to the ground-truth centre), so the
holder plays no role. Reported:
  * per-window firing rate of the spectral criterion (48-frame windows, stride 1);
  * the 'any window fires' rate per segment (the rule used in E3/E5);
  * the same for the bird evaluation set (Midway, true positions).
If the per-window rate on UAVs is low but 'any window' is high, the false
decisions are a multiple-testing artefact of the decision rule.
Output: reports/diag_fp.txt
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, WIN, band_test, frames, load, local_blobs  # noqa: E402
import a4_drones as D                                                                  # noqa: E402


def series_at_truth(frames_iter, truth):
    out = []
    for fr, (tx, ty) in frames_iter:
        bl = [b for b in local_blobs(fr, tx, ty) if np.hypot(b[0] - tx, b[1] - ty) < 25]
        b = min(bl, key=lambda q: np.hypot(q[0] - tx, q[1] - ty)) if bl else None
        out.append((b[2], b[3]) if b else (np.nan, np.nan))
    return np.array(out, float)


def window_stats(a, fps):
    fired = []
    for s in range(0, len(a) - WIN + 1):
        w = a[s:s + WIN]
        if np.isnan(w).any():
            continue
        fired.append(any((r := band_test(w[:, c], fps)) and r["fired"] for c in (0, 1)))
    return fired


def main():
    L = []
    # UAVs
    segs = D.segments()
    per_win, any_seg = [], []
    for i, idx in segs:
        fs, gt = D.load_seq(i)
        s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx]))
        k = s0 / 9.0
        im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
        it = ((cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA),
               ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k)) for j in idx)
        a = series_at_truth(it, None)
        f = window_stats(a, D.FPS)
        per_win += f; any_seg.append(any(f))
    L.append(f"UAV (DUT, 9 px, series at TRUE position): windows {len(per_win)}, per-window firing "
             f"{100 * np.mean(per_win):.1f} %, segments with any window firing {sum(any_seg)}/{len(any_seg)}")
    # birds
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    ser = {tid: [] for tid in truth}
    by = {}
    for tid, tr in truth.items():
        for k in tr:
            by.setdefault(k, []).append(tid)
    for k, fr in frames(MIDWAY, 0, max(by)):
        for tid in by.get(k, []):
            tx, ty = truth[tid][k]
            bl = [b for b in local_blobs(fr, tx, ty) if np.hypot(b[0] - tx, b[1] - ty) < 25]
            b = min(bl, key=lambda q: np.hypot(q[0] - tx, q[1] - ty)) if bl else None
            ser[tid].append((b[2], b[3]) if b else (np.nan, np.nan))
    pw, an = [], []
    for tid, s in ser.items():
        f = window_stats(np.array(s, float), 29.97)
        pw += f; an.append(any(f))
    L.append(f"Birds (Midway, series at TRUE position): windows {len(pw)}, per-window firing "
             f"{100 * np.mean(pw):.1f} %, tracks with any window firing {sum(an)}/{len(an)}")
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "diag_fp.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

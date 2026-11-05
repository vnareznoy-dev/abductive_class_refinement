#!/usr/bin/env python3
"""E20. Visibility of the bird silhouette (basis of Fig. 2 and of "99 % vs 1 %").

Reproducible version of the one-off computation of reports/support_numbers.txt (2026-10-03).
For every frame of the 50 Midway tracks the true position is known. A frame counts as
"visible" if a negative-contrast blob lies within the gate of the true position:
  WINDOW -- blobs of common.local_blobs: contrast D = G_sigma*I - I (sigma 7) in the 120x120
            attention window CENTRED ON THE TRUE POSITION, threshold max(P99 of the window, 8);
  GLOBAL -- blobs of FR.dark_blobs: the same contrast with the threshold from the full frame.
Gates: 25 px (article) and 2*S (S = median silhouette size of the track).
NOTE: this is visibility at the true position (the window is placed by the reference),
not holding: whether the holder keeps its window there is measured by the strict run measure.
Output: reports/e20_visibility.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, FR, MIDWAY, REPORTS, frames, load, local_blobs   # noqa: E402


def main():
    tr = load(os.path.join(CACHE, "eval_tracks.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tr}
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in tr}
    by = {}
    for tid, t in truth.items():
        for k in t:
            by.setdefault(k, []).append(tid)
    cnt = {tid: dict(n=0, g25=0, g2s=0, w25=0, w2s=0) for tid in truth}
    for k, fr in frames(MIDWAY, 0, max(by)):
        if k not in by:
            continue
        gb = FR.dark_blobs(fr)
        for tid in by[k]:
            tx, ty = truth[tid][k]; g2 = 2 * size[tid]
            dg = [np.hypot(b[0] - tx, b[1] - ty) for b in gb]
            dw = [np.hypot(b[0] - tx, b[1] - ty) for b in local_blobs(fr, tx, ty)]
            c = cnt[tid]; c["n"] += 1
            c["g25"] += any(d < 25 for d in dg); c["g2s"] += any(d < g2 for d in dg)
            c["w25"] += any(d < 25 for d in dw); c["w2s"] += any(d < g2 for d in dw)
    L = ["E20 visibility at the true position, 50 Midway tracks (median over tracks, min-max)", ""]
    for key, lab in (("w25", "WINDOW, gate 25 px"), ("w2s", "WINDOW, gate 2S"),
                     ("g25", "GLOBAL, gate 25 px"), ("g2s", "GLOBAL, gate 2S")):
        v = [100 * c[key] / c["n"] for c in cnt.values()]
        L.append(f"{lab:20s}: median {np.median(v):5.1f} %  min {min(v):5.1f}  max {max(v):5.1f}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e20_visibility.txt"), "w").write(t + "\n")




def uav():
    """Same visibility measure for the 24 DUT open-sky segments at 6/9/12 px (a4_drones)."""
    import cv2
    import a4_drones as D
    L = ["E20b UAV visibility at the true position (WINDOW threshold), median over segments (min)"]
    for S in D.SIZES:
        v25, v2s = [], []
        for i, idx in D.segments():
            fs, gt = D.load_seq(i)
            s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])); k = s0 / S
            if k < 1.0:
                continue
            first = cv2.imread(fs[idx[0]]); W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
            a = b = 0
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                x, y, w, h = gt[j]; tx, ty = (x + w / 2) / k, (y + h / 2) / k
                ds = [np.hypot(q[0] - tx, q[1] - ty) for q in local_blobs(fr, tx, ty)]
                a += any(d < 25 for d in ds); b += any(d < 2 * S for d in ds)
            v25.append(100 * a / len(idx)); v2s.append(100 * b / len(idx))
        L.append(f"  {S:2d} px: gate 25 px median {np.median(v25):5.1f} % (min {min(v25):.0f}) | gate 2S median {np.median(v2s):5.1f} % (min {min(v2s):.0f})")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e20_visibility.txt"), "a").write("\n" + t + "\n")


if __name__ == "__main__":
    uav() if "--uav" in sys.argv else main()

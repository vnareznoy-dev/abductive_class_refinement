#!/usr/bin/env python3
"""E21. How large are the 50 Midway birds really? The tracks were selected by the median size of
the DARK mask (D = G7*I - I > theta, formula 2-3), which leaves out the white parts of the
albatross plumage. Here, every 5th frame of each track: window 120x120 at the true position,
A = |G7*I - I| (dark AND bright deviations), threshold max(P99(A), 8), closing 5x5 to join dark and
white parts, component nearest to the true position -> max side ("full extent"). Also the dark
mask of the same frame (as in the article). Sizes in the native 1920x1080 frame; the classifier
input at imgsz 640 is 3x smaller.
Output: reports/e21_bird_extent.txt"""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, frames, load   # noqa: E402


def comp(m, close):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmin([np.hypot(c[0] - 60, c[1] - 60) for c in cen[1:]]))
    if np.hypot(cen[i][0] - 60, cen[i][1] - 60) > 25:
        return None
    return int(max(st[i][2], st[i][3]))


def main():
    tr = load(os.path.join(CACHE, "eval_tracks.pkl"))
    want = {}
    for t in tr:
        for p in t["pts"][::5]:
            want.setdefault(int(p[0]), []).append((t["id"], p))
    full, dark = {}, {}
    for k, fr in frames(MIDWAY, min(want), max(want)):
        if k not in want:
            continue
        g = cv2.copyMakeBorder(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 60, 60, 60, 60, cv2.BORDER_REPLICATE).astype(np.float32)
        for tid, p in want[k]:
            x, y = int(round(p[1])) + 60, int(round(p[2])) + 60
            w = g[y - 60:y + 60, x - 60:x + 60]; D = cv2.GaussianBlur(w, (0, 0), 7) - w
            A = np.abs(D)
            a = comp(A > 12.0, 5)          # full silhouette: |D| > 12 grey levels (P99 caps the mask at ~144 px)
            d = comp(D > max(float(np.percentile(D, 99)), 8.0), 3)
            if a: full.setdefault(tid, []).append(a)
            if d: dark.setdefault(tid, []).append(d)
    ids = sorted(set(full) & set(dark))
    mf = np.array([np.median(full[i]) for i in ids]); md = np.array([np.median(dark[i]) for i in ids])
    tsz = {t["id"]: np.median([max(q[3], q[4]) for q in t["pts"]]) for t in tr}
    mt = np.array([tsz[i] for i in ids])
    L = [f"E21 bird size, {len(ids)} tracks (median over tracks; min-max of track medians), native 1920x1080 px",
         f"  track size used in the article (dark mask, a4_select): median {np.median(mt):.0f} ({mt.min():.0f}-{mt.max():.0f})",
         f"  dark mask re-measured (window threshold):            median {np.median(md):.0f} ({md.min():.0f}-{md.max():.0f})",
         f"  full extent, dark + white parts:                     median {np.median(mf):.0f} ({mf.min():.0f}-{mf.max():.0f})",
         f"  ratio full / dark mask: median {np.median(mf / md):.1f}",
         f"  at the classifier input imgsz 640 (/3): dark {np.median(md) / 3:.1f} px, full {np.median(mf) / 3:.1f} px"]
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e21_bird_extent.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

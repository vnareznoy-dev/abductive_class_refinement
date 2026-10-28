#!/usr/bin/env python3
"""Audit of the smallest verified silhouette on real video (Midway, 5x6 px).

Reads reports/midway_tracks.pkl produced by flap_midway.py, selects the track
covering frames 1159-1215 with a 5x6 px median mask, and reports the spectral
criterion over every 48-frame window:

  (a) as tracked automatically;
  (b) with frames 1188-1190 interpolated -- in those frames a larger bird
      crosses the tracked one and the mask absorbs it (visible in
      reports/midway_track338_sheet.jpg, frame 1189: 14x15 px instead of 3-8).

Output: reports/midway_keytrack.txt
"""
import os
import pickle
import sys

import numpy as np

import flap_real as FR

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 29.97
CROSSING = (1188, 1190)


def pick_track(tracks):
    for i, t in enumerate(tracks):
        a = np.array(t)
        if int(a[0, 0]) == 1159 and int(a[-1, 0]) == 1215:
            return i, a
    raise SystemExit("key track 1159-1215 not found; run flap_midway.py first")


def report(a, label, out):
    for s in range(0, len(a) - 48 + 1):
        seg = a[s:s + 48]
        cells = []
        for lab, col in (("w", 3), ("h", 4)):
            r = FR.band_test(seg[:, col], FPS)
            cells.append(f"{lab} {r['share'] / r['base']:.2f}x @ {r['peak']:.2f} Hz"
                         f"{' FIRED' if r['fired'] else ''}")
        out.append(f"{label:24s} start {int(seg[0, 0])}: " + " | ".join(cells))


def main():
    tracks = pickle.load(open(os.path.join(HERE, "reports/midway_tracks.pkl"), "rb"))
    idx, a = pick_track(tracks)
    out = [f"track #{idx}: frames {int(a[0, 0])}-{int(a[-1, 0])}, {len(a)} points, "
           f"median mask {np.median(a[:, 3]):.0f}x{np.median(a[:, 4]):.0f} px", ""]
    report(a, "(a) as tracked", out)
    out.append("")
    b = a.copy()
    f = b[:, 0]
    keep = ~((f >= CROSSING[0]) & (f <= CROSSING[1]))
    for col in (3, 4):
        b[~keep, col] = np.interp(f[~keep], f[keep], b[keep, col])
    report(b, "(b) crossing removed", out)
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "reports/midway_keytrack.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())

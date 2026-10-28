#!/usr/bin/env python3
"""Silhouette wing-beat channel on the Midway Atoll clip (static camera).

Video: "Midway Atoll - Bird Sightings - Mar-Apr 2015.webm", Forest Starr and
Kim Starr, CC BY 2.0, Wikimedia Commons (SHA-1 9269da20ac3391d8ba45379f4bd9b5543d5e0ce3,
2641 frames, 29.97 Hz, 1920x1080). Place it as vid3/midway.webm.

Protocol (same as the article): only the upper 30 % of the frame (sky band) is
processed; multi-hypothesis tracking (track_multi.py); a track is kept if it is
>= 48 frames long and its net displacement exceeds 30 px (the camera is
static, so anything that does not translate is not a flying object); the
spectral criterion band_test (flap_real.py) is applied to 48-frame windows of
the mask width w(t) and height h(t).

Output: reports/midway_tracks.csv (one row per window) and a printed summary.
"""
from __future__ import annotations

import csv
import os
import pickle
import sys

import cv2
import numpy as np

import flap_real as FR
from track_multi import track_multi

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "vid3/midway.webm")
OUT = os.path.join(HERE, "reports/midway_tracks.csv")
MASK_BOTTOM = 0.30
MIN_LEN = 48
MIN_NET = 30.0
WIN = 48
STRIDE = 8


READ = {"n": 0}


def frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        READ["n"] += 1
        yield fr
    cap.release()


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"{VIDEO}: {n} frames, {fps:.2f} Hz")
    tracks = track_multi(frames(VIDEO), min_len=MIN_LEN, mask_bottom=MASK_BOTTOM)
    if READ["n"] < n - 5:
        raise SystemExit(f"decoded only {READ['n']} of {n} frames -- video read aborted, result invalid")
    print(f"decoded frames: {READ['n']}")
    moving = []
    for t in tracks:
        a = np.array([(p[1], p[2]) for p in t.pts], float)
        if float(np.hypot(*(a[-1] - a[0]))) > MIN_NET:
            moving.append(t)
    print(f"tracks >= {MIN_LEN} frames: {len(tracks)}; moving (net > {MIN_NET:.0f} px): {len(moving)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT.replace(".csv", ".pkl"), "wb") as fh:          # track points, for audit
        pickle.dump([[tuple(float(v) for v in p[:5]) for p in t.pts] for t in moving], fh)
    rows = []
    for ti, t in enumerate(moving):
        a = np.array([p[:5] for p in t.pts], float)          # k, cx, cy, w, h
        starts = list(range(0, len(a) - WIN + 1, STRIDE))
        spans = [(st, st + WIN) for st in starts] + ([(0, len(a))] if len(a) > WIN else [])
        for s, e in spans:
            seg = a[s:e]
            best = None
            for lab, col in (("w", 3), ("h", 4)):
                r = FR.band_test(seg[:, col], fps)
                if r and (best is None or r["share"] / r["base"] > best[1]["share"] / best[1]["base"]):
                    best = (lab, r)
            if best is None:
                continue
            lab, r = best
            rows.append(dict(track=ti, span='full' if (s, e) == (0, len(a)) and len(a) > WIN else 'win48',
                             n=len(seg), f0=int(seg[0, 0]), f1=int(seg[-1, 0]),
                             w_med=float(np.median(seg[:, 3])), h_med=float(np.median(seg[:, 4])),
                             signal=lab, peak_hz=round(r["peak"], 3),
                             ratio=round(r["share"] / r["base"], 3), fired=bool(r["fired"])))
    with open(OUT, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    fired = [r for r in rows if r["fired"]]
    print(f"windows: {len(rows)}, fired: {len(fired)} on {len({r['track'] for r in fired})} tracks")
    if fired:
        pk = np.array([r["peak_hz"] for r in fired]); rt = np.array([r["ratio"] for r in fired])
        print(f"fired peaks {pk.min():.2f}-{pk.max():.2f} Hz, ratio up to {rt.max():.2f}x")
    print("\nsmallest fired windows (median mask):")
    for r in sorted(fired, key=lambda r: r["w_med"] * r["h_med"])[:8]:
        print(f"  track {r['track']:4d} frames {r['f0']}-{r['f1']}  mask {r['w_med']:.0f}x{r['h_med']:.0f}  "
              f"{r['ratio']:.2f}x @ {r['peak_hz']:.2f} Hz ({r['signal']})")
    print("\nwindows overlapping frames 1159-1215 (article key track):")
    for r in rows:
        if r["f0"] <= 1215 and r["f1"] >= 1159:
            print(f"  track {r['track']:4d} {r['span']:5s} n={r['n']:3d} frames {r['f0']}-{r['f1']}  "
                  f"mask {r['w_med']:.0f}x{r['h_med']:.0f}  {r['ratio']:.2f}x @ {r['peak_hz']:.2f} Hz  "
                  f"{'FIRED' if r['fired'] else '-'}")
    print(f"\nreport: {OUT}")


if __name__ == "__main__":
    sys.exit(main())

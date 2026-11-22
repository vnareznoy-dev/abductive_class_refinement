#!/usr/bin/env python3
"""Article 3, answer to the reviewer (remark 3.1): tracker gaps and the 48-frame continuous series of criterion (12).

The published result takes the longest gap-free run of a track. Here gaps of up to G missing frames are filled by linear
interpolation of the box width and height, and the criterion is checked once on the longest resulting run:
  G1 - 50 tracks of the Midway bird colony: how many gaps there are, on how many tracks the criterion becomes
       applicable (run >= 48 frames) and on how many it is fulfilled, for G = 0 (published), 1, 2, 3, 5;
  G2 - the three control bird tracks for G = 0 and G = 2;
  G3 - random loss of 2, 5 and 10 % of the frames (20 repetitions per track, seed 0) followed by interpolation with
       G = 2: false firings on the continuous UAV runs and firings on the colony tracks.
Needs only data_derived/. Run: python3 rev_gaps.py   Output: reports/r8_gaps.txt"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from rev_flap import MIN_RUN, fires, wilson             # noqa: E402

DD = os.path.join(REPO, "data_derived")
OUT = os.path.join(REPO, "reports", "r8_gaps.txt")
FPS_MIDWAY, FPS_UAV = 29.97, 25.0
FPS_BIRD = {"cormorant": 25.0, "heron": 30.0, "midway_338": 29.97}


def load(fn):
    tracks = defaultdict(list)
    with open(os.path.join(DD, fn), newline="") as f:
        for r in csv.DictReader(f):
            tracks[r["track"]].append((int(r["frame"]), float(r["w"]), float(r["h"])))
    return {k: sorted(v) for k, v in tracks.items()}


def gaps(tr):
    return [b[0] - a[0] - 1 for a, b in zip(tr, tr[1:]) if b[0] - a[0] > 1]


def longest(tr, maxgap):
    """longest run in which every gap is <= maxgap frames; the gaps are filled by linear interpolation"""
    runs = [[tr[0]]]
    for a, b in zip(tr, tr[1:]):
        if b[0] - a[0] - 1 <= maxgap:
            runs[-1].append(b)
        else:
            runs.append([b])
    best = max(runs, key=lambda r: r[-1][0] - r[0][0] + 1)
    fr = np.array([x[0] for x in best]); full = np.arange(fr[0], fr[-1] + 1)
    w = np.interp(full, fr, [x[1] for x in best]); h = np.interp(full, fr, [x[2] for x in best])
    return w, h, len(full) - len(fr)


def drop(tr, p, rng):
    return [x for i, x in enumerate(tr) if i in (0, len(tr) - 1) or rng.random() > p]


def main():
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    mid = load("midway50_tracks.csv"); birds = load("bird_tracks.csv"); uav = load("uav_runs.csv")
    allg = [g for t in mid.values() for g in gaps(t)]
    P(f"G1. Midway colony, {len(mid)} tracks: {len(allg)} gaps, by length (frames: count) "
      f"{ {k: allg.count(k) for k in sorted(set(allg))} }")
    for mg in (0, 1, 2, 3, 5):
        app = fired = filled = 0; lens = []
        for t in mid.values():
            w, h, nf = longest(t, mg); lens.append(len(w))
            if len(w) >= MIN_RUN:
                app += 1; fired += int(fires(w, h, FPS_MIDWAY)[0]); filled += nf
        lo, hi = wilson(fired, app)
        P(f"  gaps up to {mg} frame(s) interpolated: criterion applicable on {app} of {len(mid)} tracks, median run "
          f"{int(np.median(lens))} frames, fulfilled on {fired} of {app} ({100 * fired / app:.0f} %, 95 % interval "
          f"{100 * lo:.0f}-{100 * hi:.0f} %), interpolated frames {filled}")
    P("G2. Control bird tracks")
    for k, t in birds.items():
        for mg in (0, 2):
            w, h, _ = longest(t, mg)
            res = fires(w, h, FPS_BIRD[k])[0] if len(w) >= MIN_RUN else "not applicable (run < 48)"
            P(f"  {k}: gaps up to {mg} interpolated, run {len(w)} frames, fulfilled: {res}")
    P(f"G3. Random frame loss + interpolation of gaps up to 2 frames (UAV runs: "
      f"{ {k: len(t) for k, t in uav.items()} } frames, no gaps in the source)")
    rng = np.random.default_rng(0)
    for p in (0.02, 0.05, 0.10):
        res = {}
        for name, tracks, fps in (("UAV", uav, FPS_UAV), ("colony", mid, FPS_MIDWAY)):
            n = k = 0
            for t in tracks.values():
                for _ in range(20):
                    w, h, _nf = longest(drop(t, p, rng), 2)
                    if len(w) >= MIN_RUN:
                        n += 1; k += int(fires(w, h, fps)[0])
            res[name] = (k, n)
        P(f"  loss {p:.0%}: UAV runs fired {res['UAV'][0]} of {res['UAV'][1]} trials; colony tracks fired "
          f"{res['colony'][0]} of {res['colony'][1]} trials ({100 * res['colony'][0] / res['colony'][1]:.0f} %)")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

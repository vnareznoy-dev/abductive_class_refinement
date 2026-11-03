#!/usr/bin/env python3
"""E4. Does holding on sub-detector evidence create false decisions?

Scene: Kormoran segment 1240-1439 (water, one bird, no other flying object).
Detector boxes (cache/det_kormoran.pkl, conf >= 0.25) are associated into
tracks exactly as in alp_on_failures.py of the Article 3 repository; every
track of >= 25 frames that is not the verified bird is a false detection on
the water. Attention is then started on each such track at its first point
and held until the end of the segment (modes: no cue / cue from the
detector's own flashes of that track). A spectral decision on the held series
is a false positive.

Output: reports/e4_specificity.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import KORMORAN, CACHE, REPORTS, WIN, FR, frames, load   # noqa: E402
import a4_experiments as X                                            # noqa: E402

SEG = (1240, 1439)
FPS = 25.0


def associate(det):
    from alp_on_failures import Trk, GATE, COAST
    tracks = []
    for k in range(SEG[0], SEG[1] + 1):
        dets = [d for d in det.get(k, []) if d[4] >= 0.25]
        taken = set()
        for t in tracks:
            if not t.alive:
                continue
            p = t.pred(); best, bj = GATE, None
            for j, d in enumerate(dets):
                if j not in taken and np.hypot(d[0] - p[0], d[1] - p[1]) < best:
                    best, bj = np.hypot(d[0] - p[0], d[1] - p[1]), j
            if bj is None:
                t.miss += 1
                t.alive = t.miss <= COAST
            else:
                taken.add(bj); d = dets[bj]
                t.vel = 0.5 * t.vel + 0.5 * (np.array(d[:2]) - np.array(t.last[:2]))
                t.last, t.miss = d, 0; t.pts[k] = d
        for j, d in enumerate(dets):
            if j not in taken:
                tracks.append(Trk(k, d))
    return [t for t in tracks if len(t.pts) >= 25]


def main():
    X.FPS_MIDWAY = FPS                       # decide() uses the module fps
    det = load(os.path.join(CACHE, "det_kormoran.pkl"))
    bird = {int(p[0]): (p[1], p[2]) for p in FR.runs(FR.track(KORMORAN, SEG[0], SEG[1] + 1)[0])[0]}
    tracks = associate(det)
    false_tr = []
    for t in tracks:
        ks = sorted(t.pts)
        is_bird = any(abs(kb - k) <= 10 and np.hypot(t.pts[k][0] - bird[kb][0], t.pts[k][1] - bird[kb][1])
                      < 30 + 3 * abs(kb - k) for k in ks for kb in [min(bird, key=lambda b: abs(b - k))])
        if not is_bird:
            false_tr.append(t)
    runs = []
    for i, t in enumerate(false_tr):
        ks = sorted(t.pts)
        start = t.pts[ks[0]][:2]
        runs.append((i, ks[0], "att_none", X.Holder(start, "none"), t))
        runs.append((i, ks[0], "att_self", X.Holder(start, "self"), t))
    for k, fr in frames(KORMORAN, SEG[0], SEG[1]):
        for i, k0, mode, h, t in runs:
            if k < k0:
                continue
            d = t.pts.get(k)
            h.step(fr, (0.0, 0.0), None if d is None else (d[0], d[1]))
    L = [f"scene: Kormoran {SEG[0]}-{SEG[1]}, truth: one bird and no other flying object",
         f"detector tracks >= 25 frames: {len(tracks)}; false (not the bird): {len(false_tr)}", ""]
    fp = 0
    for i, k0, mode, h, t in runs:
        held = np.array([not np.isnan(s[0]) for s in h.series])
        fired, _, s0 = X.decide(h.series, np.zeros(len(h.series), bool))
        fp += fired
        L.append(f"false track {i} (from frame {k0}), {mode}: held {100 * held.mean():.0f}% of "
                 f"{len(held)} frames -> {'FALSE DECISION at +' + str(s0 + WIN) if fired else 'no decision'}")
    L.append(f"\nfalse decisions: {fp} of {len(runs)} runs")
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "e4_specificity.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

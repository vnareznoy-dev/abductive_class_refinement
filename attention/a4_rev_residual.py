#!/usr/bin/env python3
"""E25 (revision of the identity article 4b, reviewer 1, remark 1). Distribution of the residual
"silhouette centre - prediction" on the REFERENCE tracks of both classes.

The prediction is that of the holder, formula (1): x^_t = x_{t-1} + v_{t-1},
v_t = (1 - beta) v_{t-1} + beta (x_t - x_{t-1}), beta = 0.4, run on the reference centres
(birds: centres of the dark silhouette masks of the 50 Midway reference tracks; UAV: centres of the
annotation boxes of the 24 DUT segments, reduced to 6 / 9 / 12 px). The body length L_t is the
median of the last 15 sizes (floor 3 px) exactly as in constraint (2); the first 5 frames of a
track are skipped (constraint not yet active), frames after a gap in the reference restart the filter.

Reports per class / size: median, 90th and 99th percentiles of the residual in px, the robust
noise scale sigma (Rayleigh: median / 1.1774), the residual relative to L_t, and the share of
frames in which the residual exceeds L_t (= frames in which the TRUE silhouette would be refuted).
Birds are also split by size bin (<5, 5-8, 8-12, >12 px).
Outputs: reports/e25_residual.txt, reports/e25_residual.json, cache/e25_residual.pkl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save   # noqa: E402
import a4_drones as D                            # noqa: E402

BETA, HIST, MIN_N, FLOOR = 0.4, 15, 5, 3.0


def residuals(xy, sz):
    """xy: list of centres (None = gap), sz: sizes. Returns list of (residual, L_t, size)."""
    out, last, vel, sizes = [], None, np.zeros(2), []
    for p, s in zip(xy, sz):
        if p is None:
            last, vel, sizes = None, np.zeros(2), []
            continue
        p = np.array(p, float)
        if last is not None:
            pred = last + vel
            if len(sizes) >= MIN_N:
                Lt = max(float(np.median(sizes[-HIST:])), FLOOR)
                out.append((float(np.hypot(*(p - pred))), Lt, float(s)))
            vel = (1 - BETA) * vel + BETA * (p - last)
        last = p
        sizes.append(float(s))
    return out


def birds():
    res = {}
    for t in load(os.path.join(CACHE, "eval_tracks.pkl")):
        pts = sorted(t["pts"]); xy, sz = [], []
        for n, p in enumerate(pts):
            if n and int(p[0]) != int(pts[n - 1][0]) + 1:
                xy.append(None); sz.append(np.nan)
            xy.append((p[1], p[2])); sz.append(max(p[3], p[4]))
        res[t["id"]] = residuals(xy, sz)
    return res


def uavs():
    res = {}
    for i, idx in D.segments():
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            xy = [((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k) for j in idx]
            sz = [max(gt[j][2], gt[j][3]) / k for j in idx]
            res[(i, idx[0], S)] = residuals(xy, sz)
    return res


def describe(rows):
    r = np.array([x[0] for x in rows]); L = np.array([x[1] for x in rows])
    return dict(n=len(r), median=float(np.median(r)), p90=float(np.percentile(r, 90)),
                p99=float(np.percentile(r, 99)), sigma=float(np.median(r) / 1.1774),
                rel_median=float(np.median(r / L)), rel_p90=float(np.percentile(r / L, 90)),
                over_L=float(100 * np.mean(r > L)), L_median=float(np.median(L)))


def main():
    B, U = birds(), uavs()
    save(dict(bird=B, uav=U), os.path.join(CACHE, "e25_residual.pkl"))
    J, out = {}, ["E25: residual |centre - prediction| on reference tracks (formula (1), beta 0.4), px", ""]
    fmt = lambda name, d: (f"{name:16s} n={d['n']:6d}  median {d['median']:5.2f}  p90 {d['p90']:5.2f}  p99 {d['p99']:6.2f}  "
                           f"sigma {d['sigma']:4.2f}  | residual/L_t median {d['rel_median']:4.2f} p90 {d['rel_p90']:4.2f} | "
                           f"> L_t {d['over_L']:5.1f} %  (L_t median {d['L_median']:4.1f})")
    allb = [x for v in B.values() for x in v]
    J["bird"] = describe(allb); out.append(fmt("birds, all", J["bird"]))
    for lo, hi in ((0, 5), (5, 8), (8, 12), (12, 999)):
        rows = [x for x in allb if lo <= x[2] < hi]
        if rows:
            J[f"bird_{lo}_{hi}"] = describe(rows); out.append(fmt(f"  size {lo}-{hi}", J[f"bird_{lo}_{hi}"]))
    for S in D.SIZES:
        rows = [x for k, v in U.items() if k[2] == S for x in v]
        J[f"uav_{S}"] = describe(rows); out.append(fmt(f"UAV {S} px", J[f"uav_{S}"]))
    out += ["", "sigma = median / 1.1774 (Rayleigh scale of a 2-D isotropic Gaussian residual)."]
    t = "\n".join(out); print(t)
    open(os.path.join(REPORTS, "e25_residual.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e25_residual.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

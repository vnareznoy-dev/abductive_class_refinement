#!/usr/bin/env python3
"""E23. Table 1 of the identity article (4б) with the size-relative tolerance r = 2L.

Re-runs E10 (a4_alp_eval: holders with and without abductive identity, birds and UAV at 6/9/12 px)
with step_all patched to store the DISTANCE from the held position to the true position instead
of the 25-px on-target flag. Everything else (holders, classifier boxes, label votes) is unchanged.
Reports, per class/size, strategy and identity on/off: share of held frames off target and the
strict run on target, for r = 25 px (must reproduce reports/e10_alp.txt) and r = 2L
(L = bird track median size / UAV equivalent size).
Outputs: cache/e23_dist.pkl, reports/e23_identity_2s.txt
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save   # noqa: E402
import a4_alp_eval as E                          # noqa: E402
import a4_drones as D                            # noqa: E402
import a4_multidet as M                          # noqa: E402
from a4_multidet_table import run_on             # noqa: E402


def step_all_dist(hs, fr, txy, dets, rec):
    X = E.X
    near = [d for d in dets if np.hypot(d[0] - txy[0], d[1] - txy[1]) < X.GATE_DET]
    fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
    for m, sd, idf, h in hs:
        h.step(fr, txy, None if fl is None else (fl[0], fl[1]), dets)
        p = h.pos[-1]
        r = rec.setdefault((m, sd, idf), {"on": [], "votes": []})
        r["on"].append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
        if m == "att_self" and p is not None:
            c = [d for d in dets if d[4] >= 0.25 and np.hypot(d[0] - p[0], d[1] - p[1]) < X.ON_TARGET]
            if c:
                r["votes"].append(E.CLS[max(c, key=lambda d: d[4])[5]])


def job(arg):
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    E.step_all = step_all_dist
    if arg == "bird":
        return E.run_birds()
    segs = D.segments()
    E.D.segments = lambda: [segs[i] for i in arg]
    return E.run_uav()


def stats(items, gate):
    """items: list of (series, distances, L); off-target share of held frames and strict run count."""
    held = off = 0; strict = 0
    for s, d, L in items:
        a = np.array(s, float); h = ~np.isnan(a[:, 0]); dd = np.array(d, float)
        g = gate(L)
        held += h.sum(); off += np.sum(h & ~(dd < g))
        strict += run_on(s, list(dd < g))
    return 100 * off / max(held, 1), strict, len(items)


def main():
    path = os.path.join(CACHE, "e23_dist.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        n = len(D.segments()); chunks = [list(range(i, n, 6)) for i in range(6)]
        R = {}
        with Pool(7) as pool:
            for r in pool.imap_unordered(job, ["bird"] + chunks):
                R.update(r); print("  job done", len(R), flush=True)
        save(R, path)
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in tracks}
    L_ = ["E23 identity article Table 1: off-target % of held frames | strict run on target (count/n, %)", ""]
    groups = [("bird", None)] + [("uav", S) for S in D.SIZES]
    for cls, S in groups:
        L_.append(f"{cls} {S or ''}")
        for m in ("att_none", "att_self", "att_ext1Hz"):
            for idf in (False, True):
                items = []
                for key, v in R.items():
                    if key[0] != cls or (cls == "uav" and key[1][2] != S):
                        continue
                    Lsz = size[key[1]] if cls == "bird" else float(S)
                    for (mm, sd, ii), (s, d, votes) in v.items():
                        if mm == m and ii == idf:
                            items.append((s, d, Lsz))
                o25, s25, n = stats(items, lambda L: 25.0)
                o2l, s2l, _ = stats(items, lambda L: 2 * L)
                lab = "ALP " if idf else "base"
                L_.append(f"  {m:10s} {lab} | 25 px: off {o25:5.1f} % strict {s25:3d}/{n} ({100 * s25 / n:4.1f} %) | "
                          f"2L: off {o2l:5.1f} % strict {s2l:3d}/{n} ({100 * s2l / n:4.1f} %)")
        L_.append("")
    t = "\n".join(L_); print(t)
    open(os.path.join(REPORTS, "e23_identity_2s.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

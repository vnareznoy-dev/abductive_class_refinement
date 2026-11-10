#!/usr/bin/env python3
"""E31 (revision of 4b, reviewer 2, remark 5). Tolerance of constraint (2) to the residual error of
platform-motion compensation.

On a pan-tilt platform the prediction (1) is moved by the image shift computed from the platform
angles; the error of this compensation remains as an apparent jump of the whole image between
frames. It is modelled as a random walk of the frame offset: o_t = o_{t-1} + e_t, e_t ~ N(0, s^2) per
axis, s = 0 (no motion), 1, 2, 4 px per frame. The frame is translated by o_t (border replicated),
the reference position moves with it; the holder (strategy S0) is not told the offset.
Variants: base (no constraint), alp (constraint (2)), alp_c2 (2 L_t). UAV of 9 and 12 px.
Output: reports/e31_jitter.txt"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save   # noqa: E402
import a4_drones as D                            # noqa: E402
from a4_multidet_table import run_on            # noqa: E402
import a4_rev_runs as RV                         # noqa: E402

SIGMAS = (0.0, 1.0, 2.0, 4.0)
SIZES = (9, 12)
VARS = {"base": RV.VARIANTS["base"], "alp": RV.VARIANTS["alp"], "alp_c2": RV.EXTRA["alp_c2"]}


def job(n):
    import cv2
    cv2.setNumThreads(1)
    i, idx = D.segments()[n]
    fs, gt = D.load_seq(i); out = {}
    for S in SIZES:
        k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
        im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
        frs = [cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA) for j in idx]
        tru = [((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k) for j in idx]
        for s in SIGMAS:
            rng = np.random.default_rng(1000 * n + S)
            off = np.cumsum(rng.normal(0, s, (len(idx), 2)), axis=0) if s > 0 else np.zeros((len(idx), 2))
            off -= off[0]
            hs = {v: RV.RevHolder(tru[0], "none", **cfg) for v, cfg in VARS.items()}
            d = {v: [] for v in VARS}
            for fr, t, o in zip(frs, tru, off):
                M = np.float32([[1, 0, o[0]], [0, 1, o[1]]])
                f2 = cv2.warpAffine(fr, M, (W, H), borderMode=cv2.BORDER_REPLICATE) if s > 0 else fr
                t2 = (t[0] + o[0], t[1] + o[1])
                for v, h in hs.items():
                    h.step(f2, t2, None, [])
                    p = h.pos[-1]
                    d[v].append(np.nan if p is None else float(np.hypot(p[0] - t2[0], p[1] - t2[1])))
            for v, h in hs.items():
                out[(n, S, s, v)] = (list(h.series), d[v])
    return out


def main():
    path = os.path.join(CACHE, "e31_jitter.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        R = {}
        with Pool(8) as pool:
            for r in pool.imap_unordered(job, range(len(D.segments()))):
                R.update(r)
        save(R, path)
    L = ["E31: residual error of platform-motion compensation (random walk of the frame, s px/frame), strategy S0, r = 2L",
         "      off = share of held frames off target, %; series = continuous >= 48 frames on target, of 24", ""]
    for S in SIZES:
        for s in SIGMAS:
            row = []
            for v in VARS:
                items = [R[k] for k in R if k[1] == S and k[2] == s and k[3] == v]
                held = off = 0; ser = 0
                for series, d in items:
                    a = np.array(series, float); h = ~np.isnan(a[:, 0]); dd = np.array(d, float); on = dd < 2 * S
                    held += h.sum(); off += np.sum(h & ~on); ser += run_on(series, list(on))
                row.append(f"{v:6s} off {100 * off / max(held, 1):5.1f} series {ser:2d}")
            L.append(f"  UAV {S:2d} px, s = {s:3.1f} px/frame | " + " | ".join(row))
        L.append("")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e31_jitter.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E24. Last column of Table 3 of the identity article (E13: start from a classifier detection)
with the tolerance r = 2L. Re-runs a4_v6_runs.Unit with the on-target flags replaced by distances
(the START condition — a detection with conf >= 0.5 within 25 px of the target — is unchanged).
Reports the share of frames on target for the reference start and the detection start, for 25 px
(must reproduce reports/v6_runs.txt, E13) and 2L. Outputs: cache/e24_dist.pkl, reports/e24_e13_2s.txt
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save   # noqa: E402
import a4_v6_runs as V                           # noqa: E402
import a4_drones as D                            # noqa: E402

X = V.X


class UnitDist(V.Unit):
    def step(self, fr, txy, dets):
        for k, h in (("tbd", self.tbd), ("sys", self.sys)):
            h.step(dets) if k == "tbd" else h.step(fr, None, None, dets)
            p = h.pos[-1]
            self.rec[k]["on"].append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
        if self.det is None:
            c = [d for d in dets if d[4] >= V.CONF_START and np.hypot(d[0] - txy[0], d[1] - txy[1]) < V.ON_T]
            if c:
                d = max(c, key=lambda d: d[4])
                self.det = X.Holder((d[0], d[1]), "self", alp=True); self.det_from = self.n
        if self.det is not None:
            self.det.step(fr, None, None, dets)
            p = self.det.pos[-1]
            self.rec["det"]["on"].append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
        self.n += 1


def job(arg):
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    V.Unit = UnitDist
    if arg == "bird":
        return V.run_birds()
    segs = D.segments()
    V.D.segments = lambda: [segs[i] for i in arg]
    return V.run_uav()


def main():
    path = os.path.join(CACHE, "e24_dist.pkl")
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
    L_ = ["E24 E13 with tolerance 2L: share of frames on target (mean over units), reference start / detection start", ""]
    for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
        keys = [k for k in R if k[0] == c and (S is None or k[1][2] == S)]
        started = [k for k in keys if R[k]["det"] is not None]
        Lsz = lambda k: size[k[1]] if c == "bird" else float(S)
        row = []
        for gname, g in (("25 px", lambda k: 25.0), ("2L", lambda k: 2 * Lsz(k))):
            on_ref = 100 * np.mean([np.mean(np.array(R[k]["sys"]["on"], float) < g(k)) for k in keys])
            on_det = 100 * np.mean([np.mean(np.array(R[k]["det"]["on"], float) < g(k)) for k in started])
            row.append(f"{gname}: {on_ref:.0f} / {on_det:.0f}")
        L_.append(f"  {c} {S or ''} started {len(started)}/{len(keys)} | " + " | ".join(row))
    t = "\n".join(L_); print(t)
    open(os.path.join(REPORTS, "e24_e13_2s.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

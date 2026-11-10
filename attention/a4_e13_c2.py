#!/usr/bin/env python3
"""E33 (revision of 4b, decision of 09.11: the limit 2 L_t becomes the main variant).
Table 3, start from a classifier detection (E13 / E24 / E30) with the holder limited by 2 L_t.

Same protocol as a4_e13_2s.py (E24): the holder starts at the first frame with a classifier box of
conf >= 0.5 within 25 px of the target; the reference-start holder runs in parallel. Both holders are
RevHolder(gate="alp", floor=3, mult=2) instead of Holder(alp=True). Distances to the reference and the
classifier votes are recorded, so the E30 subset analysis and the system answers after the detection
start (a4_v6_runs.answer_system, T = 0.5) can be computed.
Outputs: cache/e33_c2.pkl, reports/e33_e13_c2.txt"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save   # noqa: E402
import a4_v6_runs as V                           # noqa: E402
import a4_drones as D                            # noqa: E402
import a4_rev_runs as RV                         # noqa: E402

X = V.X
CFG = dict(gate="alp", floor=3.0, mult=2.0)


class UnitC2(V.Unit):
    def __init__(self, start):
        super().__init__(start)
        self.sys = RV.RevHolder(start, "self", **CFG)

    def step(self, fr, txy, dets):
        self.tbd.step(dets)
        p = self.tbd.pos[-1]
        self.rec["tbd"]["on"].append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
        for k in ("sys", "det"):
            if k == "det" and self.det is None:
                c = [d for d in dets if d[4] >= V.CONF_START and np.hypot(d[0] - txy[0], d[1] - txy[1]) < V.ON_T]
                if not c:
                    continue
                d = max(c, key=lambda d: d[4])
                self.det = RV.RevHolder((d[0], d[1]), "self", **CFG); self.det_from = self.n
            h = self.sys if k == "sys" else self.det
            h.step(fr, None, None, dets)
            p = h.pos[-1]
            self.rec[k]["on"].append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
            if p is not None and (v := V.votes_at(dets, p)):
                self.rec[k]["votes"].append(v)
        self.n += 1


def job(arg):
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    V.Unit = UnitC2
    if arg == "bird":
        return V.run_birds()
    segs = D.segments()
    V.D.segments = lambda: [segs[i] for i in arg]
    return V.run_uav()


def onfrac(d, g):
    d = np.array(d, float)
    return float(np.mean(d < g)) if len(d) else float("nan")


def main():
    path = os.path.join(CACHE, "e33_c2.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        n = len(D.segments()); chunks = [list(range(i, n, 6)) for i in range(6)]
        R = {}
        with Pool(7) as pool:
            for r in pool.imap_unordered(job, ["bird"] + chunks):
                R.update(r); print("  job done", len(R), flush=True)
        save(R, path)
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    alp = V.Alp3(); list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz("thr(det_trust, 0.5)")
    L = ["E33: Table 3, start from a detection, holder limited by 2 L_t (r = 2L unless stated)", ""]
    for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
        keys = sorted((k for k in R if k[0] == c and (S is None or k[1][2] == S)), key=str)
        st = [k for k in keys if R[k]["det"] is not None]
        g = (lambda k: 2 * size[k[1]]) if c == "bird" else (lambda k: 2.0 * S)
        ref_all = 100 * np.mean([onfrac(R[k]["sys"]["on"], g(k)) for k in keys])
        ref_st = 100 * np.mean([onfrac(R[k]["sys"]["on"], g(k)) for k in st])
        ref_after = 100 * np.mean([onfrac(R[k]["sys"]["on"][R[k]["det_from"]:], g(k)) for k in st])
        det = 100 * np.mean([onfrac(R[k]["det"]["on"], g(k)) for k in st])
        firstN = 100 * np.mean([onfrac(R[k]["sys"]["on"][:len(R[k]["det"]["on"])], g(k)) for k in st])
        ref25 = 100 * np.mean([onfrac(R[k]["sys"]["on"], 25.0) for k in keys])
        det25 = 100 * np.mean([onfrac(R[k]["det"]["on"], 25.0) for k in st])
        fps = X.FPS_MIDWAY if c == "bird" else D.FPS; truth = "bird" if c == "bird" else "drone"
        ans = [V.answer_system(R[k]["det"], fps, alp)[0] for k in st]
        wrong = sum(a not in (truth, "unknown") for a in ans)
        L.append(f"  {c} {S or ''}: started {len(st)}/{len(keys)}; on target (2L) ref start all {ref_all:.0f} %, "
                 f"started subset ref start {ref_st:.0f} %, ref start same frames after detection {ref_after:.0f} %, "
                 f"ref start first N frames {firstN:.0f} %, detection start {det:.0f} %; (25 px: ref {ref25:.0f} % / det {det25:.0f} %); "
                 f"system answers after detection start (T = 0.5): correct {ans.count(truth)}, wrong {wrong}, unknown {ans.count('unknown')}")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e33_e13_c2.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E18. Table 2 with a size-relative on-target gate (post-hoc, asked by the author).

The article's strict measure counts a frame "on target" if the held position is within
25 px of the true position. For 6-px targets this is ~4 target sizes, so a holder sitting
next to the target can count as on target (E17 on Anti-UAV: 51 % -> 37 %). Here the holders
of Table 2 are re-run exactly as in E3 (birds) and E5 (UAVs), storing the DISTANCE to the
true position, and the strict measure and the on-target bird decisions are computed for
two gates: 25 px (must reproduce the article) and 2*S (S = the unit's median target size:
bird track median size, UAV equivalent size 6/9/12 px).

Nothing in the existing caches/reports is overwritten.
Outputs: cache/e18_dist.pkl, reports/e18_gate2s.txt
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, frames, load, save   # noqa: E402
import a4_experiments as X                                     # noqa: E402
import a4_drones as D                                          # noqa: E402
import a4_multidet as M                                        # noqa: E402
from a4_multidet_table import run_on                            # noqa: E402
from a4_rules import decide                                     # noqa: E402
import a4_rules as RR                                           # noqa: E402

BIRD_MODES = ("tbd", "att_none", "att_ext1Hz")


def birds_part(tids):
    import cv2
    cv2.setNumThreads(1)
    tracks = [t for t in load(os.path.join(CACHE, "eval_tracks.pkl")) if t["id"] in tids]
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): p for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    state, dist = {}, {}
    f0, f1 = min(by_frame), max(by_frame)
    for k, fr in frames(MIDWAY, f0, f1):
        for tid in by_frame.get(k, []):
            p = truth[tid][k]; tx, ty = p[1], p[2]
            near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < X.GATE_DET]
            flash = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
            flash_xy = None if flash is None else (flash[0], flash[1])
            if tid not in state:
                state[tid] = [m for m in X.modes_for((tx, ty)) if m[0] in BIRD_MODES]
            for mode, sd, obj in state[tid]:
                if mode == "tbd":
                    obj.step(det.get(k, []))
                else:
                    obj.step(fr, (tx, ty), flash_xy, det.get(k, []))
                ps = obj.pos[-1]
                dist.setdefault((tid, mode, sd), []).append(None if ps is None else float(np.hypot(ps[0] - tx, ps[1] - ty)))
    return {key: (list(obj.series), dist[(tid, mode, sd)]) for tid, lst in state.items() for mode, sd, obj in lst
            for key in [(tid, mode, sd)]}


def bird_native():
    """SZV at 1920 px, threshold 0.1 (as a4_figs_v3.tbd_native) with distances."""
    det = load(os.path.join(CACHE, "det_midway_1920.pkl")); tr = load(os.path.join(CACHE, "eval_tracks.pkl"))
    X.FLASH = 0.1; out = {}
    for t in tr:
        pts = t["pts"]; o = X.TBD((pts[0][1], pts[0][2])); ds = []
        for p in pts:
            o.step(det.get(int(p[0]), [])); ps = o.pos[-1]
            ds.append(None if ps is None else float(np.hypot(ps[0] - p[1], ps[1] - p[2])))
        out[(t["id"], "tbd1920", None)] = (list(o.series), ds)
    X.FLASH = 0.25
    return out


_MODEL = None


def _init():
    global _MODEL
    import cv2, torch
    torch.set_num_threads(1); cv2.setNumThreads(1)
    from ultralytics import YOLO
    _MODEL = YOLO(D.WEIGHTS)


def uav_one(job):
    """Re-run a4_drones.run_segment with distances: patch ON_T to +inf and record distances."""
    i, idx, S = job
    import math, cv2
    fs, gt = D.load_seq(i)
    s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])); k = s0 / S
    if k < 1.0:
        return {}
    first = cv2.imread(fs[idx[0]])
    W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
    imgsz = max(64, int(math.ceil(max(W, H) / 32.0) * 32))
    c0 = np.array([gt[idx[0]][0] + gt[idx[0]][2] / 2, gt[idx[0]][1] + gt[idx[0]][3] / 2]) / k
    holders = [("tbd", None, X.TBD(tuple(c0))), ("att_none", None, X.Holder(tuple(c0), "none"))]
    for sd in D.SEEDS:
        holders.append(("att_ext1Hz", sd, X.Holder(tuple(c0), "ext", period=int(D.FPS), sigma=15.0, seed=sd)))
    dist = {}
    for j in idx:
        fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
        x, y, w, h = gt[j]; tx, ty = (x + w / 2) / k, (y + h / 2) / k
        r = _MODEL.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0]
        dets = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                 float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        for mode, sd, obj in holders:
            if mode == "tbd":
                obj.step(dets)
            else:
                obj.step(fr, (tx, ty), None, dets)
            ps = obj.pos[-1]
            dist.setdefault((mode, sd), []).append(None if ps is None else float(np.hypot(ps[0] - tx, ps[1] - ty)))
    return {(i, idx[0], S, mode, sd): (list(obj.series), dist[(mode, sd)]) for mode, sd, obj in holders}


def on_of(ds, gate):
    return [d is not None and d < gate for d in ds]


def pc(k, n):
    lo, hi = M.wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f} %, {lo:.0f}-{hi:.0f})"


def main():
    path = os.path.join(CACHE, "e18_dist.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
        tids = [t["id"] for t in tracks]
        parts = [tids[n::8] for n in range(8)]
        jobs = [(i, idx, S) for i, idx in D.segments() for S in D.SIZES]
        R = {"bird": {}, "uav": {}}
        with Pool(8) as pool:
            for r in pool.imap_unordered(birds_part, parts):
                R["bird"].update(r); print(f"  bird part done ({len(R['bird'])})", flush=True)
        R["bird"].update(bird_native())
        with Pool(8, initializer=_init) as pool:
            for n, r in enumerate(pool.imap_unordered(uav_one, jobs)):
                R["uav"].update(r); print(f"  uav {n + 1}/{len(jobs)}", flush=True)
        save(R, path)
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in tracks}
    L = ["E18 Table 2 with on-target gate 25 px (article) and 2*S (S = median target size); post-hoc", ""]
    L.append("BIRDS: strict run on target | on-target bird decision (rule full)")
    for m in ("tbd", "tbd1920", "att_none", "att_ext1Hz"):
        it = [(k, v) for k, v in R["bird"].items() if k[1] == m]
        for lab, g in (("25px", lambda k: 25.0), ("2S", lambda k: 2 * size[k[0]])):
            st = sum(run_on(s, on_of(d, g(k))) for k, (s, d) in it)
            dec = sum(decide(s, on_of(d, g(k)), RR.FPS_BIRD, "full")[1] for k, (s, d) in it)
            L.append(f"  {m:11s} {lab:5s} strict {pc(st, len(it))}  decision on target {pc(dec, len(it))}")
    L.append("")
    L.append("UAVs: strict run on target | false bird (rule full, independent of gate)")
    for m in ("tbd", "att_none", "att_ext1Hz"):
        for S in D.SIZES:
            it = [(k, v) for k, v in R["uav"].items() if k[3] == m and k[2] == S]
            if not it:
                continue
            a = sum(run_on(s, on_of(d, 25.0)) for k, (s, d) in it)
            b = sum(run_on(s, on_of(d, 2.0 * S)) for k, (s, d) in it)
            fb = sum(decide(s, on_of(d, 25.0), D.FPS, "full")[0] for k, (s, d) in it)
            L.append(f"  {m:11s} {S:2d} px  25px {pc(a, len(it))}  2S {pc(b, len(it))}  false bird {fb}/{len(it)}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e18_gate2s.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

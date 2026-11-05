#!/usr/bin/env python3
"""E22. Working range of the method versus target size («the UAV flies away»).

The 24 DUT open-sky segments of E5 are reduced to L = 4, 6, 9, 12, 16, 24, 32 px (k = L0/L,
INTER_AREA, as in E5). For every segment and L, per frame:
  classifier : c_t (formula 1, YOLO weights of the article, conf >= 0.05, imgsz of the reduced frame);
  visibility : a blob of the window threshold (formulas 2-3, common.local_blobs, window at the true
               position) within r = 2L of the true position;
  holding    : tbd (СЗВ) and att_none (У0) exactly as a4_drones.run_segment, distance to the truth.
Per segment: coverage = share of frames with c_t >= 0.25; visibility share; strict run on target
(longest gap-free run >= 48 frames, >= 50 % within r = 2L). At L = 6/9/12 the numbers must match
E18 (У0 12/33/83, СЗВ 4/17/25 %), E20 (visibility 25/87/100 %) and Table 1 (coverage).
Outputs: cache/e22_range.pkl, reports/e22_range.txt
"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, local_blobs, save   # noqa: E402
import a4_drones as D                                        # noqa: E402
import a4_experiments as X                                   # noqa: E402
import a4_multidet as M                                      # noqa: E402
from a4_multidet_table import run_on                          # noqa: E402

SIZES = (4, 6, 9, 12, 16, 24, 32)
_MODEL = None


def _init():
    global _MODEL
    import cv2, torch
    torch.set_num_threads(1); cv2.setNumThreads(1)
    from ultralytics import YOLO
    _MODEL = YOLO(D.WEIGHTS)


def job(arg):
    import cv2
    i, idx, L = arg
    fs, gt = D.load_seq(i)
    L0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])); k = L0 / L
    if k < 1.0:
        return None
    first = cv2.imread(fs[idx[0]])
    W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
    imgsz = max(64, int(math.ceil(max(W, H) / 32.0) * 32))
    c0 = (np.array([gt[idx[0]][0] + gt[idx[0]][2] / 2, gt[idx[0]][1] + gt[idx[0]][3] / 2]) / k)
    hold = [("tbd", X.TBD(tuple(c0))), ("att_none", X.Holder(tuple(c0), "none"))]
    conf, vis, dist = [], [], {m: [] for m, _ in hold}
    for j in idx:
        fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
        x, y, w, h = gt[j]; tx, ty = (x + w / 2) / k, (y + h / 2) / k
        r = _MODEL.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0]
        dets = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                 float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        conf.append(max((d[4] for d in dets if np.hypot(d[0] - tx, d[1] - ty) < D.GATE_DET), default=0.0))
        vis.append(any(np.hypot(b[0] - tx, b[1] - ty) < 2 * L for b in local_blobs(fr, tx, ty)))
        for m, obj in hold:
            obj.step(dets) if m == "tbd" else obj.step(fr, (tx, ty), None, dets)
            ps = obj.pos[-1]
            dist[m].append(float("nan") if ps is None else float(np.hypot(ps[0] - tx, ps[1] - ty)))
    out = dict(seg=(i, idx[0]), L=L, L0=L0, W=W, H=H, cov=float(np.mean(np.array(conf) >= 0.25)), vis=float(np.mean(vis)))
    for m, obj in hold:
        out[f"strict_{m}"] = bool(run_on(list(obj.series), [d < 2 * L for d in dist[m]]))
    return out


def main():
    path = os.path.join(CACHE, "e22_range.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        jobs = [(i, idx, L) for i, idx in D.segments() for L in SIZES]
        R = []
        with Pool(8, initializer=_init) as pool:
            for n, r in enumerate(pool.imap_unordered(job, jobs)):
                if r:
                    R.append(r)
                if n % 20 == 0:
                    print(f"  {n + 1}/{len(jobs)}", flush=True)
        save(R, path)
    L_ = ["E22 working range vs target size L (24 DUT segments; median over segments for shares of frames; strict = % of segments)", ""]
    for L in SIZES:
        rs = [r for r in R if r["L"] == L]
        st = {m: sum(r[f"strict_{m}"] for r in rs) for m in ("tbd", "att_none")}
        lo, hi = M.wilson(st["att_none"], len(rs))
        L_.append(f"  L={L:2d} n={len(rs)}: frame median {int(np.median([r['W'] for r in rs]))}x{int(np.median([r['H'] for r in rs]))} | "
                  f"coverage c>=0.25 median {100 * np.median([r['cov'] for r in rs]):3.0f} % | visibility median {100 * np.median([r['vis'] for r in rs]):3.0f} % | "
                  f"strict СЗВ {100 * st['tbd'] / len(rs):3.0f} %  У0 {100 * st['att_none'] / len(rs):3.0f} % ({lo:.0f}-{hi:.0f})")
    t = "\n".join(L_); print(t)
    open(os.path.join(REPORTS, "e22_range.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

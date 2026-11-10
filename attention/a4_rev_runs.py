#!/usr/bin/env python3
"""E26 (revision of the identity article 4b). One pass over the data with several variants of the
association rule of the attention window; everything else (window, threshold, cues, classifier
boxes, data, scoring) is unchanged. Protocol fixed before the run.

Variants (each with the strategies S0 = att_none, SC = att_self, SE-1 = att_ext1Hz x 5 seeds):
  base      basic holding: nearest silhouette within the 30 px window gate (no constraint (2));
  alp       abductive identity, constraint (2): limit L_t = median of the last 15 accepted sizes,
            floor 3 px (the published method; must reproduce cache/e23_dist.pkl);
  alp_f4,   constraint (2) with a lower bound of L_t tied to the centre noise (reviewer 1, remark 2):
  alp_f6    L_t = max(median size, F), F = 4 and 6 px (about 3 and 4 sigma of the reference residual,
            sigma = 1.4 px for birds, reports/e25_residual.txt);
  fix10,    nearest neighbour with a FIXED gate radius 10 / 20 px and the missed-detection
  fix20     hypothesis (unseen), i.e. constraint (2) without the size (reviewer 2, remark 1);
  stat      statistical gate (reviewer 2, remark 1): radius sqrt(chi2_2(0.99)) * sigma_hat = 3.03 sigma_hat,
            sigma_hat = RMS per axis of the residuals of the last 15 accepted silhouettes, minimum 1 px,
            30 px until 5 residuals are available; missed-detection hypothesis as in alp;
  b02, b06  alp with velocity smoothing beta = 0.2 / 0.6 (published 0.4)   (reviewer 1, remark 7);
  k3, k8    alp with coast length K = 3 / 8 frames (published 5)           (reviewer 1, remark 7).
Recorded per holder and frame: distance of the accepted silhouette to the reference (nan if none),
distance of the track position including coast frames (nan only when lost), state, and for every
accepted silhouette in the hold state the residual to the prediction and the limit in force;
for SC the classifier votes (conf >= 0.25 within 25 px of the held position) as in E7/E10.
Outputs: cache/e26_runs.pkl (analysis in a4_rev_tables.py)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, frames, load, local_blobs, save   # noqa: E402
import a4_drones as D                                                 # noqa: E402
import a4_experiments as X                                            # noqa: E402

CLS = {0: "bird", 1: "drone"}
VARIANTS = {
    "base": dict(gate="base"),
    "alp": dict(gate="alp", floor=3.0),
    "alp_f4": dict(gate="alp", floor=4.0),
    "alp_f6": dict(gate="alp", floor=6.0),
    "fix10": dict(gate="fixed", radius=10.0),
    "fix20": dict(gate="fixed", radius=20.0),
    "stat": dict(gate="stat"),
    "b02": dict(gate="alp", floor=3.0, beta=0.2),
    "b06": dict(gate="alp", floor=3.0, beta=0.6),
    "k3": dict(gate="alp", floor=3.0, coast=3),
    "k8": dict(gate="alp", floor=3.0, coast=8),
}
EXTRA = {   # E26b: limit c * L_t (added after E26 showed that the floor does not change the bird series)
    "alp_c15": dict(gate="alp", floor=3.0, mult=1.5),
    "alp_c2": dict(gate="alp", floor=3.0, mult=2.0),
}
C2SENS = {   # E26c: sensitivity of the 2 L_t limit to beta and K (after the decision to make 2 L_t the main variant)
    "c2_b02": dict(gate="alp", floor=3.0, mult=2.0, beta=0.2),
    "c2_b06": dict(gate="alp", floor=3.0, mult=2.0, beta=0.6),
    "c2_k3": dict(gate="alp", floor=3.0, mult=2.0, coast=3),
    "c2_k8": dict(gate="alp", floor=3.0, mult=2.0, coast=8),
}
ACTIVE = (C2SENS if os.environ.get("E26_C2SENS") else
          EXTRA if os.environ.get("E26_EXTRA") else VARIANTS)   # worker processes re-import the module
CHI = float(np.sqrt(9.2103))      # chi-square, 2 d.o.f., 0.99


class RevHolder(X.Holder):
    def __init__(self, start, cue_mode, gate="base", floor=3.0, radius=30.0, beta=0.4, coast=X.COAST, mult=1.0, **kw):
        super().__init__(start, cue_mode, alp=gate != "base", **kw)
        self.gate, self.floor, self.radius, self.beta, self.coast, self.mult = gate, floor, radius, beta, coast, mult
        self.res = []        # residuals of accepted silhouettes in the hold state (for stat and the report)
        self.reslog = []     # (residual, limit) per accepted hold silhouette
        self.states = []

    def limit(self):
        if self.gate == "alp":
            return self.mult * max(float(np.median(self.sizes)), self.floor) if len(self.sizes) >= X.ID_MIN else 30.0
        if self.gate == "fixed":
            return self.radius
        if self.gate == "stat":
            if len(self.res) < X.ID_MIN:
                return 30.0
            r = np.array(self.res[-X.ID_HIST:])
            return max(CHI * float(np.sqrt(np.mean(r ** 2) / 2)), 1.0)
        return 30.0

    def step(self, fr, truth_xy, flash_xy, dets=()):
        if self.cue_mode == "ext" and self.n % self.period == 0:
            self.cue = np.array(truth_xy) + self.rng.normal(0, self.sigma, 2)
        elif self.cue_mode == "self":
            if self.last is not None:
                anchor, gate = self.last + self.vel, X.GATE_DET
            elif self.lost_xy is not None:
                anchor = self.lost_xy + self.vel * self.since
                gate = min(X.GATE_DET + 2.0 * self.since, 150.0)
            else:
                anchor = None
            if anchor is not None:
                c = [d for d in dets if d[4] >= X.FLASH and np.hypot(d[0] - anchor[0], d[1] - anchor[1]) < gate]
                if c:
                    d = min(c, key=lambda d: np.hypot(d[0] - anchor[0], d[1] - anchor[1]))
                    self.cue = np.array((d[0], d[1]))
        self.n += 1
        pick, pred, state, lim = None, None, None, None
        if self.last is not None:
            pred = self.last + self.vel; state = "hold"
            bl = [b for b in local_blobs(fr, pred[0], pred[1]) if np.hypot(b[0] - pred[0], b[1] - pred[1]) < 30.0]
            if bl:
                if self.gate == "base":
                    pick = min(bl, key=lambda b: np.hypot(b[0] - pred[0], b[1] - pred[1]))
                else:
                    lim = self.limit()
                    ok = [b for b in bl if np.hypot(b[0] - pred[0], b[1] - pred[1]) <= lim]
                    pick = min(ok, key=lambda b: np.hypot(b[0] - pred[0], b[1] - pred[1])) if ok else None
            if pick is not None:
                r = float(np.hypot(pick[0] - pred[0], pick[1] - pred[1]))
                self.res.append(r); self.reslog.append((r, lim if lim is not None else 30.0, len(self.series)))
        if pick is None and self.cue is not None and (self.last is None or self.miss >= self.coast):
            bl = [b for b in local_blobs(fr, self.cue[0], self.cue[1])
                  if np.hypot(b[0] - self.cue[0], b[1] - self.cue[1]) < 45.0]
            if bl:
                pick = min(bl, key=lambda b: np.hypot(b[0] - self.cue[0], b[1] - self.cue[1]))
                self.vel = np.zeros(2); state = "reacq"
        if pick is not None:
            if self.last is not None:
                self.vel = (1 - self.beta) * self.vel + self.beta * (np.array(pick[:2]) - self.last)
            self.last = np.array(pick[:2]); self.miss = 0
            self.lost_xy, self.since = None, 0
            self.series.append((pick[2], pick[3])); self.pos.append(tuple(pick[:2]))
            if self.gate != "base":
                self.sizes = (self.sizes + [max(pick[2], pick[3])])[-X.ID_HIST:]
        else:
            self.miss += 1
            if self.last is not None and self.miss <= self.coast:
                self.last = self.last + self.vel; state = "coast"
            else:
                state = "lost"
                if self.last is not None:
                    self.lost_xy, self.since = self.last.copy(), 0
                self.last = None
                self.since += 1
            self.series.append((np.nan, np.nan)); self.pos.append(None)
        self.states.append(state)


def holders(start, fps):
    out = []
    for v, cfg in ACTIVE.items():
        out.append(("att_none", None, v, RevHolder(start, "none", **cfg)))
        out.append(("att_self", None, v, RevHolder(start, "self", **cfg)))
        for sd in X.SEEDS:
            out.append(("att_ext1Hz", sd, v, RevHolder(start, "ext", period=int(fps), sigma=15.0, seed=sd, **cfg)))
    return out


def step_all(hs, fr, txy, dets, rec):
    near = [d for d in dets if np.hypot(d[0] - txy[0], d[1] - txy[1]) < X.GATE_DET]
    fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
    for m, sd, v, h in hs:
        h.step(fr, txy, None if fl is None else (fl[0], fl[1]), dets)
        p = h.pos[-1]
        r = rec.setdefault((m, sd, v), {"d": [], "dc": [], "votes": []})
        r["d"].append(np.nan if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))
        r["dc"].append(np.nan if h.last is None else float(np.hypot(h.last[0] - txy[0], h.last[1] - txy[1])))
        if m == "att_self" and p is not None:
            c = [d for d in dets if d[4] >= 0.25 and np.hypot(d[0] - p[0], d[1] - p[1]) < X.ON_TARGET]
            if c:
                b = max(c, key=lambda d: d[4])
                r["votes"].append((CLS[b[5]], float(b[4])))


def pack(hs, rec):
    return {(m, sd, v): dict(series=list(h.series), states=list(h.states), reslog=list(h.reslog), **rec[(m, sd, v)])
            for m, sd, v, h in hs}


def run_birds(ids):
    tracks = [t for t in load(os.path.join(CACHE, "eval_tracks.pkl")) if t["id"] in ids]
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    hs, rec = {}, {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            txy = truth[tid][k]
            if tid not in hs:
                hs[tid] = holders(txy, X.FPS_MIDWAY)
            step_all(hs[tid], fr, txy, det.get(k, []), rec.setdefault(tid, {}))
    return {("bird", tid): pack(hs[tid], rec[tid]) for tid in hs}


def run_uav(seg_ids):
    import cv2
    det = load(os.path.join(CACHE, "e6_det_uav_ours.pkl")); out = {}
    segs = D.segments()
    for n in seg_ids:
        i, idx = segs[n]
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
            c0 = ((gt[idx[0]][0] + gt[idx[0]][2] / 2) / k, (gt[idx[0]][1] + gt[idx[0]][3] / 2) / k)
            hs, rec = holders(c0, D.FPS), {}
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                txy = ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k)
                step_all(hs, fr, txy, det[(i, idx[0], S, j)], rec)
            out[("uav", (i, idx[0], S))] = pack(hs, rec)
    return out


def job(arg):
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    t0 = time.time()
    r = run_birds(arg[1]) if arg[0] == "bird" else run_uav(arg[1])
    print(f"  {arg[0]} job {len(arg[1])} units done in {time.time() - t0:.0f} s", flush=True)
    return r


def main():
    name = ("e26c_runs.pkl" if os.environ.get("E26_C2SENS") else
            "e26b_runs.pkl" if os.environ.get("E26_EXTRA") else "e26_runs.pkl")
    path = os.path.join(CACHE, name)
    ids = [t["id"] for t in load(os.path.join(CACHE, "eval_tracks.pkl"))]
    n = len(D.segments())
    jobs = [("bird", ids[i::4]) for i in range(4)] + [("uav", list(range(i, n, 4))) for i in range(4)]
    R = {}
    with Pool(8) as pool:
        for r in pool.imap_unordered(job, jobs):
            R.update(r)
    save(R, path); print("saved", path, len(R))


if __name__ == "__main__":
    main()

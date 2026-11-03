#!/usr/bin/env python3
"""Article 4, draft v6: experiments 1-3 (protocol fixed before the run).

Data: 50 Midway bird tracks (classifier input 640, true class bird) and 24 DUT open-sky segments
at 6/9/12 px (true class drone). Classifier: the Article 3 bird/drone detector. The window is
Holder(cue "self", alp=True): attention window + abductive association (a4_track_alp.pl).

E11 -- the classifier asks ALP when it is not sure.
  Classifier alone (baseline): tracking-by-detection; its answer for the target is the class of
    its most confident box on the track if that confidence >= 0.9, otherwise the 70 % majority of
    >= 5 boxes, otherwise 'unknown'.
  System: if the classifier has a box with confidence >= 0.9 on the held target, its class is the
    answer. Otherwise the question goes to the ALP of Article 3 (air_objects.pl, unchanged) with
    only the facts that are groundable at range:
      f_det(C, P)  -- majority class C of the classifier votes on the held track (>= 5 votes),
                      P = median confidence of those votes;
      f_flap(B)    -- criterion (6) on the longest held run, asserted only if the run has >= 48 frames.
    Speed, hover and turn facts are NOT asserted (not measurable in pixels without range).
    ALP answer: classify(C, Justification) -> confirmed / revised / accepted / kinematic_only / unknown.
E12 -- ALP association for the classifier's own tracking: tracking-by-detection accepts a box only
  if its centre is within one body length (median size of the last 15 accepted boxes, floor 3 px,
  from 5 boxes on) of the prediction; three classifiers (ours, YOLOv8n, YOLOv8s COCO).
E13 -- start from a classifier detection: the holder starts at the first frame in which the
  classifier has a box with confidence >= 0.5 whose centre is within 25 px of the target (the
  reference picks WHICH box is the assigned target, the start position is the box centre).
  Units in which this never happens count as not started.
Output: reports/v6_runs.txt, reports/v6_runs.json, cache/v6_runs.pkl
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPO, REPORTS, WIN, band_test, frames, load, save   # noqa: E402
import a4_drones as D                                                                 # noqa: E402
import a4_experiments as X                                                            # noqa: E402
from a4_multidet import wilson                                                        # noqa: E402
from a4_multidet_table import run_on                                                  # noqa: E402

CLS = {0: "bird", 1: "drone"}
CONF_SURE, CONF_START, CONF_VOTE, ON_T = 0.9, 0.5, 0.25, 25.0


class AlpTBD(X.TBD):
    """Tracking-by-detection with the ALP kinematic constraint (E12)."""

    def __init__(self, start, alp):
        super().__init__(start); self.alp = alp; self.sizes = []

    def step(self, dets):
        if not self.alp:
            return super().step(dets)
        pick = None
        if self.last is not None:
            pred = self.last + self.vel
            lim = max(float(np.median(self.sizes)), X.ALP_FLOOR) if len(self.sizes) >= X.ID_MIN else X.GATE_DET
            c = [d for d in dets if d[4] >= X.FLASH and np.hypot(d[0] - pred[0], d[1] - pred[1]) < min(lim, X.GATE_DET)]
            if c:
                pick = min(c, key=lambda d: np.hypot(d[0] - pred[0], d[1] - pred[1]))
        if pick is not None:
            self.vel = 0.6 * self.vel + 0.4 * (np.array(pick[:2]) - self.last)
            self.last = np.array(pick[:2]); self.miss = 0
            self.sizes = (self.sizes + [max(pick[2], pick[3])])[-X.ID_HIST:]
            self.series.append((pick[2], pick[3])); self.pos.append(tuple(pick[:2]))
        else:
            self.miss += 1
            if self.last is not None and self.miss <= X.COAST:
                self.last = self.last + self.vel
            else:
                self.last = None
            self.series.append((np.nan, np.nan)); self.pos.append(None)


def votes_at(dets, p):
    c = [d for d in dets if d[4] >= CONF_VOTE and np.hypot(d[0] - p[0], d[1] - p[1]) < ON_T]
    if not c:
        return None
    d = max(c, key=lambda d: d[4]); return (CLS[d[5]], d[4])


class Unit:
    """One target: classifier-alone TBD, system with reference start, system with detection start."""

    def __init__(self, start):
        self.tbd = X.TBD(start); self.sys = X.Holder(start, "self", alp=True)
        self.det = None; self.det_from = None
        self.rec = {k: {"on": [], "votes": []} for k in ("tbd", "sys", "det")}
        self.n = 0

    def step(self, fr, txy, dets):
        for k, h in (("tbd", self.tbd), ("sys", self.sys)):
            if k == "tbd":
                h.step(dets)
            else:
                h.step(fr, None, None, dets)
            p = h.pos[-1]
            self.rec[k]["on"].append(p is not None and np.hypot(p[0] - txy[0], p[1] - txy[1]) < ON_T)
            if p is not None and (v := votes_at(dets, p)):
                self.rec[k]["votes"].append(v)
        if self.det is None:
            c = [d for d in dets if d[4] >= CONF_START and np.hypot(d[0] - txy[0], d[1] - txy[1]) < ON_T]
            if c:
                d = max(c, key=lambda d: d[4])
                self.det = X.Holder((d[0], d[1]), "self", alp=True); self.det_from = self.n
        if self.det is not None:
            self.det.step(fr, None, None, dets)
            p = self.det.pos[-1]
            self.rec["det"]["on"].append(p is not None and np.hypot(p[0] - txy[0], p[1] - txy[1]) < ON_T)
            if p is not None and (v := votes_at(dets, p)):
                self.rec["det"]["votes"].append(v)
        self.n += 1

    def dump(self):
        out = {"n": self.n, "det_from": self.det_from}
        for k, h in (("tbd", self.tbd), ("sys", self.sys), ("det", self.det)):
            out[k] = None if h is None else dict(series=list(h.series), **self.rec[k])
        return out


def run_birds():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl")); det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    units = {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            txy = truth[tid][k]
            units.setdefault(tid, Unit(txy)).step(fr, txy, det.get(k, []))
        if k % 500 == 0:
            print("birds frame", k, flush=True)
    return {("bird", tid): u.dump() for tid, u in units.items()}


def run_uav():
    det = load(os.path.join(CACHE, "e6_det_uav_ours.pkl")); out = {}
    for n, (i, idx) in enumerate(D.segments()):
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
            u = Unit(((gt[idx[0]][0] + gt[idx[0]][2] / 2) / k, (gt[idx[0]][1] + gt[idx[0]][3] / 2) / k))
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                u.step(fr, ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k), det[(i, idx[0], S, j)])
            out[("uav", (i, idx[0], S))] = u.dump()
        print("uav segment", n + 1, flush=True)
    return out


# ------------------------------------------------------------------ E11 decision
def longest_run(series):
    a = np.array(series, float); ok = ~np.isnan(a[:, 0]); best = cur = 0; end = -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    return a[end - best + 1:end + 1] if best >= WIN else None


class Alp3:
    """The ALP of Article 3 (air_objects.pl), unchanged."""

    def __init__(self):
        from pyswip import Prolog
        self.pl = Prolog(); self.pl.consult(os.path.join(REPO, "air_objects.pl"))

    def classify(self, det, flap):
        for f in ("f_det(_,_)", "f_flap(_)", "f_speed(_)", "f_speed_cv(_)", "f_hover(_)", "f_turn_irreg(_)",
                  "f_straight(_)", "f_turn_bias(_)"):
            list(self.pl.query(f"retractall({f})"))
        if det is not None:
            self.pl.assertz(f"f_det({det[0]},{det[1]})")
        if flap is not None:
            self.pl.assertz(f"f_flap({'true' if flap else 'false'})")
        r = list(self.pl.query("classify(C, J)", maxresult=1))[0]
        return str(r["C"]), str(r["J"]).split("(")[0]


def answer_alone(votes):
    if votes and max(c for _, c in votes) >= CONF_SURE:
        return max(votes, key=lambda v: v[1])[0], "classifier"
    if len(votes) >= 5:
        for c in ("drone", "bird"):
            if sum(v == c for v, _ in votes) / len(votes) >= 0.7:
                return c, "classifier"
    return "unknown", "classifier"


def answer_system(rec, fps, alp):
    votes = rec["votes"]
    if votes and max(c for _, c in votes) >= CONF_SURE:
        return max(votes, key=lambda v: v[1])[0], "classifier"
    det = None
    if len(votes) >= 5:
        cls = max(("bird", "drone"), key=lambda c: sum(v == c for v, _ in votes))
        det = (cls, float(np.median([c for v, c in votes if v == cls])))
    run = longest_run(rec["series"])
    flap = None if run is None else any((r := band_test(run[:, c], fps)) and r["fired"] for c in (0, 1))
    c, how = alp.classify(det, flap)
    return c, "alp:" + how


def main():
    path = os.path.join(CACHE, "v6_runs.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        R = {**run_uav(), **run_birds()}; save(R, path)
    E10 = load(os.path.join(CACHE, "e10_series.pkl"))
    same = all(np.array_equal(np.array(R[k]["sys"]["series"], float), np.array(E10[k][("att_self", None, True)][0], float), equal_nan=True)
               for k in R)
    L = [f"system run reproduces the E10 ALP holder (US): {same}", ""]; J = {"reproduced": same}
    alp = Alp3()
    groups = [("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)]
    # ---------------- E11
    L.append("E11: classifier alone vs classifier that asks ALP (correct / wrong / unknown; d->b = drone called bird)")
    for c, S in groups:
        keys = [k for k in R if k[0] == c and (S is None or k[1][2] == S)]
        truth = "bird" if c == "bird" else "drone"; fps = X.FPS_MIDWAY if c == "bird" else D.FPS
        res = {}
        for name, fn in (("alone", lambda r: answer_alone(r["tbd"]["votes"])), ("system", lambda r: answer_system(r["sys"], fps, alp))):
            ans = [fn(R[k]) for k in keys]
            labs = [a for a, _ in ans]; how = [h for _, h in ans]
            cor = labs.count(truth); unk = labs.count("unknown"); wr = len(labs) - cor - unk
            d2b = labs.count("bird") if c == "uav" else 0
            asked = sum(h.startswith("alp") for h in how)
            kinds = {h: how.count(h) for h in sorted(set(how))}
            res[name] = dict(n=len(labs), correct=cor, wrong=wr, unknown=unk, d2b=d2b, asked_alp=asked, kinds=kinds)
            L.append(f"  {c} {S or ''} {name:6s} n={len(labs):2d} correct {cor:2d} wrong {wr:2d} unknown {unk:2d}"
                     + (f" d->b {d2b}" if c == 'uav' else "") + f"  asked ALP {asked}  {kinds}")
        J[f"e11_{c}_{S}"] = res
    # ---------------- E12
    L.append(""); L.append("E12: ALP association for the classifier's own tracking (strict run on target %, frames on target %)")
    for c, S in groups:
        for cfg in ("ours", "v8n", "v8s"):
            for alp_on in (False, True):
                items = []
                if c == "bird":
                    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
                    det = load(os.path.join(CACHE, "det_midway.pkl" if cfg == "ours" else f"e6_det_bird_{cfg}_640.pkl"))
                    for t in tracks:
                        pts = t["pts"]; tb = AlpTBD((pts[0][1], pts[0][2]), alp_on); on = []
                        for p in pts:
                            tb.step(det.get(int(p[0]), [])); q = tb.pos[-1]
                            on.append(q is not None and np.hypot(q[0] - p[1], q[1] - p[2]) < ON_T)
                        items.append((tb.series, on))
                else:
                    det = load(os.path.join(CACHE, f"e6_det_uav_{cfg}.pkl"))
                    for k in [k for k in R if k[0] == "uav" and k[1][2] == S]:
                        i, f0, _ = k[1]; fs, gt = D.load_seq(i)
                        idx = [j for (ii, ff, ss, j) in det if (ii, ff, ss) == (i, f0, S)]
                        idx.sort(); kk = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
                        c0 = ((gt[idx[0]][0] + gt[idx[0]][2] / 2) / kk, (gt[idx[0]][1] + gt[idx[0]][3] / 2) / kk)
                        tb = AlpTBD(c0, alp_on); on = []
                        for j in idx:
                            tb.step(det[(i, f0, S, j)]); q = tb.pos[-1]
                            tx, ty = (gt[j][0] + gt[j][2] / 2) / kk, (gt[j][1] + gt[j][3] / 2) / kk
                            on.append(q is not None and np.hypot(q[0] - tx, q[1] - ty) < ON_T)
                        items.append((tb.series, on))
                n = len(items); kk_ = sum(run_on(s, o) for s, o in items); lo, hi = wilson(kk_, n)
                held = sum(int((~np.isnan(np.array(s, float)[:, 0])).sum()) for s, o in items)
                off = sum(int((~np.isnan(np.array(s, float)[:, 0]) & ~np.array(o)).sum()) for s, o in items)
                ontime = 100 * np.mean([np.mean(o) for s, o in items])
                J[f"e12_{c}_{S}_{cfg}_{'alp' if alp_on else 'base'}"] = dict(n=n, strict=100 * kk_ / n, ci=[lo, hi],
                                                                             off=100 * off / max(1, held), on_time=ontime)
                L.append(f"  {c} {S or ''} {cfg:4s} {'TBD+ALP' if alp_on else 'TBD    '} strict {100 * kk_ / n:5.1f} ({lo:3.0f}-{hi:3.0f})"
                         f"  on target {ontime:4.1f} % of frames  off-target {100 * off / max(1, held):4.1f} % of held")
    # ---------------- E13
    L.append(""); L.append("E13: start from a classifier detection (conf >= 0.5) vs start from the reference")
    for c, S in groups:
        keys = [k for k in R if k[0] == c and (S is None or k[1][2] == S)]
        started = [k for k in keys if R[k]["det"] is not None]
        delay = [R[k]["det_from"] for k in started]
        fps = X.FPS_MIDWAY if c == "bird" else D.FPS
        s_ref = sum(run_on(R[k]["sys"]["series"], R[k]["sys"]["on"]) for k in keys)
        s_det = sum(run_on(R[k]["det"]["series"], R[k]["det"]["on"]) for k in started)
        on_ref = 100 * np.mean([np.mean(R[k]["sys"]["on"]) for k in keys])
        on_det = 100 * np.mean([np.mean(R[k]["det"]["on"]) for k in started]) if started else float("nan")
        ans = [answer_system(R[k]["det"], fps, alp)[0] for k in started]
        truth = "bird" if c == "bird" else "drone"
        J[f"e13_{c}_{S}"] = dict(n=len(keys), started=len(started), median_delay=float(np.median(delay)) if delay else None,
                                 strict_ref=s_ref, strict_det=s_det, on_ref=on_ref, on_det=on_det,
                                 correct_det=ans.count(truth), d2b=ans.count("bird") if c == "uav" else 0)
        L.append(f"  {c} {S or ''} started {len(started)}/{len(keys)} (median delay {np.median(delay) if delay else '-'} frames); "
                 f"strict run on target ref start {s_ref}/{len(keys)}, detection start {s_det}/{len(started)}; "
                 f"on target {on_ref:.0f} % vs {on_det:.0f} % of frames; system answers after detection start: "
                 f"correct {ans.count(truth)}" + (f", d->b {ans.count('bird')}" if c == 'uav' else ""))
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "v6_runs.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "v6_runs.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

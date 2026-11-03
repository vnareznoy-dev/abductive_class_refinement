#!/usr/bin/env python3
"""E10. Abductive association of the attention window (a4_track_alp.pl), protocol fixed before the run.

Holder(alp=True): among the silhouettes in the window gate, the target is the nearest one whose
distance to the prediction does not exceed one body length (median size of the last 15 accepted
silhouettes, floor 3 px); if all are refuted, the target is "unseen" in this frame (coast).
The Python decisions are cross-checked against the Prolog theory on a sample of frames.

(Below: the E9 description of the measures, which are identical.)

Holder(identity=True) accepts a silhouette only if its size max(w,h) is within 0.5-2.0 and its
contrast within 0.4-2.5 of the medians of the last 15 accepted silhouettes (from 5 samples on);
otherwise the frame is a miss (coast); re-acquisition from a cue is checked the same way
(a4_experiments.ID_*). Every strategy is run with identity False and True on the same data;
the False runs must reproduce the series of the main run (e3 / e5 / e6 caches).

Measures: strict run on target (>= 48 frames, >= 50 % on target), share of held frames off
target, criterion decisions (birds: on target / other object; UAV: false 'bird'), and for US
(classifier cue) the classifier votes on the held position + the soaring rule 'bank' of E8
(threshold = max over training UAV, grouped 5-fold CV) -> labels.
Output: reports/e9_identity.txt, reports/e9_identity.json, cache/e9_series.pkl
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, frames, load, save   # noqa: E402
import a4_drones as D                                            # noqa: E402
import a4_experiments as X                                       # noqa: E402
import a4_rules as RR                                            # noqa: E402
import a4_soar as SO                                             # noqa: E402
from a4_attention_fig import same                                # noqa: E402
from a4_multidet import wilson                                   # noqa: E402
from a4_multidet_table import run_on                             # noqa: E402

CLS = {0: "bird", 1: "drone"}
SAMPLE = []   # ALP decisions (prediction, candidates, body, chosen) for the Prolog cross-check


def holders(start, fps):
    out = []
    for idf in (False, True):
        out.append(("att_none", None, idf, X.Holder(start, "none", alp=idf)))
        out.append(("att_self", None, idf, X.Holder(start, "self", alp=idf)))
        for sd in X.SEEDS:
            out.append(("att_ext1Hz", sd, idf, X.Holder(start, "ext", period=int(fps), sigma=15.0, seed=sd, alp=idf)))
    return out


def step_all(hs, fr, txy, dets, rec):
    near = [d for d in dets if np.hypot(d[0] - txy[0], d[1] - txy[1]) < X.GATE_DET]
    fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
    for m, sd, idf, h in hs:
        h.step(fr, txy, None if fl is None else (fl[0], fl[1]), dets)
        p = h.pos[-1]
        r = rec.setdefault((m, sd, idf), {"on": [], "votes": []})
        r["on"].append(p is not None and np.hypot(p[0] - txy[0], p[1] - txy[1]) < X.ON_TARGET)
        if m == "att_self" and p is not None:
            c = [d for d in dets if d[4] >= 0.25 and np.hypot(d[0] - p[0], d[1] - p[1]) < X.ON_TARGET]
            if c:
                r["votes"].append(CLS[max(c, key=lambda d: d[4])[5]])


def run_birds():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl")); det = load(os.path.join(CACHE, "det_midway.pkl"))
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
        if k % 500 == 0:
            print("birds frame", k, flush=True)
    for tid in list(hs)[:20]:
        SAMPLE.extend(next(h for m, sd, idf, h in hs[tid] if m == "att_none" and idf).alp_log[::5])
    return {("bird", tid): {(m, sd, idf): (list(h.series), rec[tid][(m, sd, idf)]["on"], rec[tid][(m, sd, idf)]["votes"])
                            for m, sd, idf, h in hs[tid]} for tid in hs}


def run_uav():
    det = load(os.path.join(CACHE, "e6_det_uav_ours.pkl")); out = {}
    for n, (i, idx) in enumerate(D.segments()):
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
            SAMPLE.extend(next(h for m, sd, idf, h in hs if m == "att_none" and idf).alp_log[::7])
            out[("uav", (i, idx[0], S))] = {(m, sd, idf): (list(h.series), rec[(m, sd, idf)]["on"], rec[(m, sd, idf)]["votes"])
                                             for m, sd, idf, h in hs}
        print("uav segment", n + 1, flush=True)
    return out


def label_votes(v):
    if len(v) < 5:
        return "unknown"
    for c in ("drone", "bird"):
        if v.count(c) / len(v) >= 0.7:
            return c
    return "unknown"


def prolog_check(sample):
    from pyswip import Prolog
    pl = Prolog(); pl.consult(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a4_track_alp.pl"))
    agree = 0
    for pred, cands, body, chosen in sample:
        for q in ("f_pred(_,_)", "f_cand(_,_,_,_)", "f_body(_)"):
            list(pl.query(f"retractall({q})"))
        pl.assertz(f"f_pred({float(pred[0])},{float(pred[1])})")
        for n, (x, y, sz) in enumerate(cands):
            pl.assertz(f"f_cand({n},{float(x)},{float(y)},{float(sz)})")
        if body is not None:
            pl.assertz(f"f_body({float(body)})")
        r = list(pl.query("explain(H, _)", maxresult=1))[0]["H"]
        got = None if str(r) == "unseen" else int(str(r)[5:-1])
        agree += got == chosen
    return f"Prolog theory agrees with the Python decisions on {agree}/{len(sample)} sampled frames"


def main():
    path = os.path.join(CACHE, "e10_series.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        R = {**run_uav(), **run_birds()}
        save(R, path); save(SAMPLE, os.path.join(CACHE, "e10_alp_sample.pkl"))
    # reproduction check of the identity=False runs
    B, U, S6 = (load(os.path.join(CACHE, f)) for f in ("e3_series.pkl", "e5_series.pkl", "e6_series.pkl"))
    chk = []
    for (c, unit), v in R.items():
        for (m, sd, idf), (s, o, vo) in v.items():
            if idf:
                continue
            if c == "bird":
                chk.append(same(s, B[(unit, m, sd)][0]))
            elif m == "att_self":
                chk.append(same(s, S6[("uav", unit, "ours", "att_self")][0]))
            else:
                chk.append(same(s, U[unit][(m, sd)][0]))
    L = [f"alp=False reproduces the main run: {all(chk)} ({sum(chk)}/{len(chk)})"]
    L.append(prolog_check(load(os.path.join(CACHE, "e10_alp_sample.pkl")))); L.append("")
    J = {"reproduced": all(chk)}
    groups = [("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)]
    for c, S in groups:
        L.append(f"{c} {S or ''}".strip())
        for m in ("att_none", "att_self", "att_ext1Hz"):
            for idf in (False, True):
                items = [(s, o) for (cc, unit), v in R.items() if cc == c and (S is None or unit[2] == S)
                         for (mm, sd, i2), (s, o, vo) in v.items() if mm == m and i2 == idf]
                n = len(items); k = sum(run_on(s, o) for s, o in items); lo, hi = wilson(k, n)
                held = sum(int(~np.isnan(a)) for s, o in items for a in np.array(s, float)[:, 0])
                off = sum(int((~np.isnan(np.array(s, float)[:, 0]) & ~np.array(o)).sum()) for s, o in items)
                fps = RR.FPS_BIRD if c == "bird" else RR.FPS_UAV
                dec = [RR.decide(s, o, fps, "full") for s, o in items]
                d_on = sum(x[1] for x in dec); d_any = sum(x[0] for x in dec)
                key = f"{c}_{S}_{m}_{'alp' if idf else 'base'}"
                J[key] = dict(n=n, strict=100 * k / n, ci=[lo, hi], off=100 * off / max(1, held), held=held,
                              dec_on=d_on, dec_other=d_any - d_on)
                L.append(f"  {m:10s} {'ALP' if idf else 'base'} strict {100 * k / n:5.1f} ({lo:3.0f}-{hi:3.0f})  "
                         f"off-target {100 * off / max(1, held):4.1f} % of {held:5d} held  "
                         + (f"decisions on target {d_on}, other {d_any - d_on}" if c == "bird" else f"false 'bird' {d_any}"))
        L.append("")
    # labels for US: votes + criterion priority + bank rule (E8), base vs identity
    for idf in (False, True):
        U_ = []
        for (c, unit), v in R.items():
            s, o, vo = v[("att_self", None, idf)]
            run = SO.longest(s)
            U_.append(dict(cls=c, grp=("b", unit) if c == "bird" else ("u", unit[0]), vote=label_votes(vo),
                           run=run is not None,
                           crit=SO.fired(run, SO.FPS_B if c == "bird" else SO.FPS_U) if run is not None else False,
                           bank=SO.feats(run)["bank"] if run is not None else np.nan))
        gl = sorted({u["grp"] for u in U_}); rng = np.random.default_rng(0); rng.shuffle(gl)
        fold = {g: n % 5 for n, g in enumerate(gl)}
        for u in U_:
            tr = [x["bank"] for x in U_ if x["cls"] == "uav" and fold[x["grp"]] != fold[u["grp"]] and x["run"]]
            u["rule"] = bool(u["run"] and tr and u["bank"] > max(tr))
        tag = "ALP " if idf else "base"
        for name, f in (("v1 fixed rule", lambda x: "bird" if x["crit"] and x["vote"] != "drone" else "drone" if (not x["crit"] and x["vote"] == "drone") else "unknown"),
                        ("priority + bank", lambda x: "bird" if (x["crit"] or x["rule"]) else "drone" if x["vote"] == "drone" else "unknown")):
            b = [f(x) for x in U_ if x["cls"] == "bird"]; d = [f(x) for x in U_ if x["cls"] == "uav"]
            J[f"labels_{tag.strip()}_{name}"] = dict(bird=dict(bird=b.count("bird"), drone=b.count("drone"), unknown=b.count("unknown")),
                                                     uav=dict(drone=d.count("drone"), bird=d.count("bird"), unknown=d.count("unknown")))
            L.append(f"labels US {tag} {name:16s} birds: bird {b.count('bird'):2d} drone {b.count('drone'):2d} unknown {b.count('unknown'):2d} | "
                     f"UAV: drone {d.count('drone'):2d} bird {d.count('bird')} unknown {d.count('unknown'):2d}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e10_alp.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e10_alp.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

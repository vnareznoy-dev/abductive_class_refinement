#!/usr/bin/env python3
"""E8. Knowledge-light rules for SOARING birds (protocol fixed before the run).

Physical priors (no training data):
  F1 'bank'  -- a soaring bird banks in turns that last seconds; its silhouette elongation
               changes slowly. F1 = std of the 15-frame moving average of log(w/h) over the
               run (slow component only: pixel quantisation and wing beats are faster).
               A multirotor keeps its shape.
  F2 'elong' -- median of max(w,h)/min(w,h): spread wings make the silhouette elongated.
Trajectory features are NOT used: the DUT camera may follow the vehicle, the Midway camera is
fixed, so image-plane trajectories differ by recording, not by class.

Part 1, validity: reference series (blob at the true position every frame) of the 50 birds
  (cache/truth_series_birds.pkl) and of the 72 UAV runs (24 segments x 6/9/12 px, built here
  the same way). ROC AUC of each feature, birds vs UAV, and soaring birds (criterion (6) not
  fired on the reference run) vs UAV.
Part 2, in the system: held series of variant B (window + classifier cue, cache/e6_series.pkl,
  classifier 'ours', birds imgsz 640) on the longest held run >= 48 frames, classifier votes of
  E7 (reports/e7_assist.csv). Rule: 'bird' if F > tau, tau = maximum of F over the TRAINING UAV
  runs (no training UAV would be called a bird); grouped cross-validation, groups = bird track /
  DUT sequence, 5 folds. Fused label v2: 'bird' if criterion or rule fired, 'drone' if neither
  fired and votes say drone, otherwise 'unknown'.
Output: reports/e8_soar.txt, reports/e8_soar.json
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, WIN, band_test, load, local_blobs, save   # noqa: E402
import a4_drones as D                                                        # noqa: E402

FPS_B, FPS_U = 29.97, 25.0


def longest(a):
    a = np.asarray(a, float); ok = ~np.isnan(a[:, 0]); best = cur = 0; end = -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    return a[end - best + 1:end + 1] if best >= WIN else None


def feats(run):
    w, h = run[:, 0], run[:, 1]
    la = np.log(np.maximum(w, 1) / np.maximum(h, 1))
    slow = np.convolve(la, np.ones(15) / 15, mode="valid")
    return {"bank": float(np.std(slow)), "elong": float(np.median(np.maximum(w, h) / np.maximum(np.minimum(w, h), 1)))}


def fired(run, fps):
    return any((r := band_test(run[:, c], fps)) and r["fired"] for c in (0, 1))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def uav_truth_series():
    path = os.path.join(CACHE, "e8_truth_series_uav.pkl")
    if os.path.exists(path):
        return load(path)
    out = {}
    for i, idx in D.segments():
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
            ser = []
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                tx, ty = (gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k
                bl = [b for b in local_blobs(fr, tx, ty) if np.hypot(b[0] - tx, b[1] - ty) < 25]
                b = min(bl, key=lambda q: np.hypot(q[0] - tx, q[1] - ty)) if bl else None
                ser.append((b[2], b[3]) if b else (np.nan, np.nan))
            out[(i, idx[0], S)] = ser
    save(out, path)
    return out


def main():
    L, J = [], {}
    # ---------------- part 1: validity on reference series
    TB = load(os.path.join(CACHE, "truth_series_birds.pkl")); TU = uav_truth_series()
    b = [(feats(r), fired(r, FPS_B)) for r in (longest(s) for s in TB.values()) if r is not None]
    u = [feats(r) for r in (longest(s) for s in TU.values()) if r is not None]
    L.append(f"Part 1, reference series: birds {len(b)} (soaring = criterion not fired: {sum(not f for _, f in b)}), UAV runs {len(u)}")
    for f in ("bank", "elong"):
        a_all = auc([x[f] for x, _ in b], [x[f] for x in u])
        a_soar = auc([x[f] for x, fr in b if not fr], [x[f] for x in u])
        qb = np.percentile([x[f] for x, fr in b if not fr], [25, 50, 75]); qu = np.percentile([x[f] for x in u], [25, 50, 75])
        L.append(f"  {f:6s} AUC birds vs UAV {a_all:.2f}, soaring birds vs UAV {a_soar:.2f}; quartiles soaring birds "
                 f"{qb.round(3).tolist()}, UAV {qu.round(3).tolist()}")
        J[f"p1_{f}"] = dict(auc_all=a_all, auc_soar=a_soar)
    # ---------------- part 2: in the system (variant B)
    S6 = load(os.path.join(CACHE, "e6_series.pkl"))
    votes = {}
    for r in csv.DictReader(open(os.path.join(REPORTS, "e7_assist.csv"))):
        if r["cls"] == "bird" and r["res"] == "640":
            votes[("bird", int(r["unit"]))] = r["B_label"]
        elif r["cls"] == "drone":
            i, f0 = r["unit"].split(":"); votes[("uav", (int(i), int(f0), int(r["res"])))] = r["B_label"]
    units = []
    for key, (s, o) in S6.items():
        c, unit, cfg, mode = key
        if mode != "att_self" or (c == "bird" and cfg != ("ours", 640)) or (c == "uav" and cfg != "ours"):
            continue
        run = longest(s)
        grp = ("b", unit) if c == "bird" else ("u", unit[0])
        vk = ("bird", unit) if c == "bird" else ("uav", unit)
        units.append(dict(cls=c, unit=unit, grp=grp, vote=votes[vk], run=run is not None,
                          crit=fired(run, FPS_B if c == "bird" else FPS_U) if run is not None else False,
                          **(feats(run) if run is not None else {"bank": np.nan, "elong": np.nan})))
    groups = sorted({u_["grp"] for u_ in units}); rng = np.random.default_rng(0); rng.shuffle(groups)
    fold = {g: n % 5 for n, g in enumerate(groups)}
    L.append(""); L.append(f"Part 2, held series of variant B: birds {sum(u_['cls'] == 'bird' for u_ in units)}, "
                           f"UAV runs {sum(u_['cls'] == 'uav' for u_ in units)}; units with a run >= 48: "
                           f"{sum(u_['run'] for u_ in units)}")
    for f in ("bank", "elong"):
        for u_ in units:
            tr = [x[f] for x in units if x["cls"] == "uav" and fold[x["grp"]] != fold[u_["grp"]] and x["run"]]
            u_["rule_" + f] = bool(u_["run"] and tr and u_[f] > max(tr))
        res = {}
        for c in ("bird", "uav"):
            rs = [x for x in units if x["cls"] == c]
            v1 = [("bird" if x["crit"] and x["vote"] != "drone" else "drone" if (not x["crit"] and x["vote"] == "drone") else "unknown") for x in rs]
            v2 = [("bird" if (x["crit"] or x["rule_" + f]) else "drone" if x["vote"] == "drone" else "unknown") for x in rs]
            res[c] = dict(n=len(rs), rule_fired=sum(x["rule_" + f] for x in rs),
                          v1={k: v1.count(k) for k in ("bird", "drone", "unknown")},
                          v2={k: v2.count(k) for k in ("bird", "drone", "unknown")})
        J[f"p2_{f}"] = res
        L.append(f"  rule '{f}': birds rule fired {res['bird']['rule_fired']}/{res['bird']['n']}, UAV rule fired "
                 f"{res['uav']['rule_fired']}/{res['uav']['n']} (= drones that would be called birds)")
        L.append(f"     birds  fused v1 {res['bird']['v1']}  ->  v2 {res['bird']['v2']}")
        L.append(f"     UAV    fused v1 {res['uav']['v1']}  ->  v2 {res['uav']['v2']}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e8_soar.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e8_soar.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

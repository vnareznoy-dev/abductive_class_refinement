#!/usr/bin/env python3
"""E7. Does the attention method help the classifier? (protocol fixed before the run)

Units: the 50 Midway bird tracks (true class 'bird') and the 24 DUT open-sky segments at
6 / 9 / 12 px (true class 'drone'). The target is assigned once from the reference position
(as in single-object tracking benchmarks); afterwards the reference is used ONLY for scoring.
Classifier: the Article 3 bird/drone detector (rf/runs/bird_drone_clean).

Variants
  A  classifier alone: full frame (birds: imgsz 640 and 1920; UAV: reduced frame at its own
     size), tracking-by-detection (a4_experiments.TBD); votes = classes of the boxes the track
     takes.
  B  classifier + window: a4_experiments.Holder with the classifier flashes of the full frame
     as cue (strategy US); votes = classifier boxes with conf >= 0.25 within 25 px of the HELD
     position.
  C  classifier + window + crop: as B, but the classifier is run on the 120 x 120 crop of the
     window (around the prediction), upsampled to 640 x 640.
  Reference (not a system): A_oracle -- votes from all boxes within 25 px of the TRUE position
     in the full frame, i.e. the classifier with perfect association.

Measures (per unit)
  visibility -- share of frames where the classifier input of the variant contains a box with
                conf >= 0.25 within 25 px of the true position (A, B: full frame; C: crop);
  label      -- from votes only:   'drone'/'bird' if >= 5 votes and >= 70 % for that class,
                                   otherwise 'unknown';
  fused      -- votes + criterion (6) on the longest held run (a4_pipeline rule):
                'bird' if criterion fired and votes do not say drone; 'drone' if criterion did
                not fire and votes say drone; otherwise 'unknown'.
  Scored against the true class: correct / wrong / unknown; 'drone labelled bird' is reported apart.
Output: reports/e7_assist.csv, reports/e7_summary.txt, cache/e7_series.pkl
"""
import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, WEIGHTS, WIN, band_test, frames, load, save   # noqa: E402
import a4_drones as D                                                                     # noqa: E402
import a4_experiments as X                                                                # noqa: E402
from a4_pipeline import ROI, ZOOM, boxes                                                  # noqa: E402

MIN_VOTES, SHARE, CONF_V, ON_T = 5, 0.7, 0.25, 25.0
CLS = {0: "bird", 1: "drone"}


def label_votes(votes):
    n = len(votes)
    if n < MIN_VOTES:
        return "unknown"
    for c in ("drone", "bird"):
        if sum(v == c for v in votes) / n >= SHARE:
            return c
    return "unknown"


def criterion(series, fps):
    a = np.array(series, float); ok = ~np.isnan(a[:, 0]); best = cur = 0; end = -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    if best < WIN:
        return False
    run = a[end - best + 1:end + 1]
    return any((r := band_test(run[:, c], fps)) and r["fired"] for c in (0, 1))


def fuse(votes_label, fired):
    if fired and votes_label != "drone":
        return "bird"
    if not fired and votes_label == "drone":
        return "drone"
    return "unknown"


def near(dets, xy, conf=CONF_V):
    c = [d for d in dets if d[4] >= conf and np.hypot(d[0] - xy[0], d[1] - xy[1]) < ON_T]
    return max(c, key=lambda d: d[4]) if c else None


class Unit:
    def __init__(self, start):
        self.tbd = X.TBD(start)
        self.hb = X.Holder(start, "self")
        self.hc = X.Holder(start, "self")
        self.v = {k: [] for k in ("A", "B", "C", "Aor")}
        self.vis = {k: [] for k in ("A", "C")}

    def step(self, fr, txy, full, crop_fn):
        self.tbd.step(full)                                  # A
        p = self.tbd.pos[-1]
        if p is not None:
            d = min(full, key=lambda d: np.hypot(d[0] - p[0], d[1] - p[1]))
            self.v["A"].append(CLS[d[5]])
        d = near(full, txy)                                  # A_oracle + visibility of the full frame
        self.vis["A"].append(d is not None)
        if d is not None:
            self.v["Aor"].append(CLS[d[5]])
        self.hb.step(fr, None, None, full)                   # B
        p = self.hb.pos[-1]
        if p is not None and (d := near(full, p)) is not None:
            self.v["B"].append(CLS[d[5]])
        h = self.hc                                          # C
        anchor = h.last + h.vel if h.last is not None else (h.lost_xy + h.vel * h.since if h.lost_xy is not None else None)
        crop = crop_fn(fr, anchor) if anchor is not None else []
        self.vis["C"].append(near(crop, txy) is not None)
        h.step(fr, None, None, crop)
        p = h.pos[-1]
        if p is not None and (d := near(crop, p)) is not None:
            self.v["C"].append(CLS[d[5]])

    def result(self, fps):
        out = {}
        for k in ("A", "Aor", "B", "C"):
            lab = label_votes(self.v[k])
            if k == "A":
                fired = criterion(self.tbd.series, fps)
            elif k == "B":
                fired = criterion(self.hb.series, fps)
            elif k == "C":
                fired = criterion(self.hc.series, fps)
            else:
                fired = False
            out[k] = dict(votes=len(self.v[k]), label=lab, fused=fuse(lab, fired), fired=fired)
        out["vis_A"] = float(np.mean(self.vis["A"])); out["vis_C"] = float(np.mean(self.vis["C"]))
        return out


def make_crop_fn(model):
    def crop_fn(fr, anchor):
        H, W = fr.shape[:2]
        r = min(ROI, H, W)
        x0 = int(np.clip(anchor[0] - r / 2, 0, W - r)); y0 = int(np.clip(anchor[1] - r / 2, 0, H - r))
        up = cv2.resize(fr[y0:y0 + r, x0:x0 + r], (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)
        return boxes(model.predict(up, conf=0.05, imgsz=ZOOM, device="mps", verbose=False)[0], ZOOM / r, x0, y0)
    return crop_fn


def birds(model, res):
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    det = load(os.path.join(CACHE, "det_midway.pkl" if res == 640 else "det_midway_1920.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    units, crop_fn = {}, make_crop_fn(model)
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            txy = truth[tid][k]
            u = units.setdefault(tid, Unit(txy))
            u.step(fr, txy, det.get(k, []), crop_fn)
        if k % 500 == 0:
            print("birds", res, "frame", k, flush=True)
    return [dict(cls="bird", unit=tid, res=res, **flat(u.result(X.FPS_MIDWAY))) for tid, u in units.items()]


def uavs(model):
    det = load(os.path.join(CACHE, "e6_det_uav_ours.pkl"))
    rows, crop_fn = [], make_crop_fn(model)
    for n, (i, idx) in enumerate(D.segments()):
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            if k < 1:
                continue
            im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
            c0 = ((gt[idx[0]][0] + gt[idx[0]][2] / 2) / k, (gt[idx[0]][1] + gt[idx[0]][3] / 2) / k)
            u = Unit(c0)
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                txy = ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k)
                u.step(fr, txy, det[(i, idx[0], S, j)], crop_fn)
            rows.append(dict(cls="drone", unit=f"{i}:{idx[0]}", res=S, **flat(u.result(D.FPS))))
        print("uav segment", n + 1, flush=True)
    return rows


def flat(r):
    o = {"vis_A": r["vis_A"], "vis_C": r["vis_C"]}
    for k in ("A", "Aor", "B", "C"):
        for f in ("votes", "label", "fused", "fired"):
            o[f"{k}_{f}"] = r[k][f]
    return o


def summary(rows):
    L = ["E7: does the attention method help the classifier? (correct / wrong / unknown, %; 'd->b' = drone labelled bird)", ""]
    for cls, res in [("bird", 640), ("bird", 1920), ("drone", 6), ("drone", 9), ("drone", 12)]:
        rs = [r for r in rows if r["cls"] == cls and r["res"] == res]
        if not rs:
            continue
        n = len(rs)
        L.append(f"{cls} {res}: n={n}  visibility median full frame {100 * np.median([r['vis_A'] for r in rs]):.0f} %, "
                 f"window crop {100 * np.median([r['vis_C'] for r in rs]):.0f} %")
        for k, name in (("Aor", "classifier, perfect association (reference)"), ("A", "A classifier + TBD"),
                        ("B", "B classifier + window"), ("C", "C classifier + window + crop")):
            for f in ("label", "fused"):
                if k == "Aor" and f == "fused":
                    continue
                labs = [r[f"{k}_{f}"] for r in rs]
                cor = sum(l == cls for l in labs); unk = sum(l == "unknown" for l in labs); wr = n - cor - unk
                d2b = sum(l == "bird" for l in labs) if cls == "drone" else 0
                L.append(f"   {name:46s} {f:5s} correct {100 * cor / n:5.1f}  wrong {100 * wr / n:5.1f}  unknown {100 * unk / n:5.1f}"
                         + (f"  d->b {d2b}" if cls == "drone" else "") + f"  votes median {np.median([r[f'{k}_votes'] for r in rs]):.0f}")
        L.append("")
    return "\n".join(L)


def main():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    rows = uavs(model) + birds(model, 640) + birds(model, 1920)
    with open(os.path.join(REPORTS, "e7_assist.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    t = summary(rows); print(t)
    open(os.path.join(REPORTS, "e7_summary.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

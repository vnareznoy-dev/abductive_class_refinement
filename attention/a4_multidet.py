#!/usr/bin/env python3
"""E6. Holding with different classifiers and reference trackers.

Classifiers (no retraining, weights as published or as trained for Article 3):
  ours  -- two-class bird/drone detector of Article 3 (rf/runs/bird_drone_clean);
  v8n   -- YOLOv8n, COCO (80 classes, incl. bird / airplane / kite), Ultralytics;
  v8s   -- YOLOv8s, COCO.
A detection is any box of any class (as in E1), conf >= 0.05 is cached.

Holding strategies that DEPEND on the classifier:
  tbd      -- tracking-by-detection of a4_experiments (gate 45 px, coast 5);
  bytetrack, botsort -- reference implementations of Ultralytics with their default
              configuration files (bytetrack.yaml, botsort.yaml; no ReID); the target
              is the first track whose box lies within 45 px of the true position
              (truth is used once, as for all holders); the held series is the box
              of the detection matched to that track id in the frame;
  att_self -- attention window with the classifier flash as cue (a4_experiments.Holder).
Strategies that do NOT use the classifier (att_none, att_ext1Hz) are taken from the
main run (cache/e3_series.pkl, cache/e5_series.pkl).

Data: the 50 Midway bird tracks (imgsz 640 and 1920) and the 24 DUT open-sky UAV
segments at 6 / 9 / 12 px (detector at the native size of the reduced frame).
Usage: python3 a4_multidet.py det   -- detection caches (cache/e6_det_*.pkl)
       python3 a4_multidet.py eval  -- reports/e6_multidet.csv, reports/e6_summary.txt,
                                       cache/e6_series.pkl
"""
import csv
import math
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, WEIGHTS, WIN, frames, load, save   # noqa: E402
import a4_experiments as X                                                      # noqa: E402
import a4_drones as D                                                           # noqa: E402
import a4_rules as RR                                                           # noqa: E402

BF = os.path.expanduser("~/flying-justice-bf")
MODELS = {"ours": WEIGHTS, "v8n": os.path.join(BF, "yolov8n.pt"), "v8s": os.path.join(BF, "yolov8s.pt")}
BIRD_RES = (640, 1920)


def boxes(r):
    return [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
             float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
             float(b.conf[0]), int(b.cls[0])) for b in r.boxes]


# ------------------------------------------------------------------ detections
def det_birds(name, res):
    out = os.path.join(CACHE, f"e6_det_bird_{name}_{res}.pkl")
    if os.path.exists(out):
        return
    if name == "ours":     # reuse the caches of the main run
        src = "det_midway.pkl" if res == 640 else "det_midway_1920.pkl"
        save(load(os.path.join(CACHE, src)), out); print("reused", src, "->", out, flush=True); return
    from ultralytics import YOLO
    model = YOLO(MODELS[name])
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    need = {int(p[0]) for t in tracks for p in t["pts"]}
    d = {}
    for k, fr in frames(MIDWAY, min(need), max(need)):
        if k in need:
            d[k] = boxes(model.predict(fr, conf=0.05, imgsz=res, device="mps", verbose=False)[0])
            if len(d) % 500 == 0:
                print(name, res, len(d), "/", len(need), flush=True)
    save(d, out); print("saved", out, flush=True)


def reduced(i, idx, S):
    fs, gt = D.load_seq(i)
    k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
    if k < 1.0:
        return None
    first = cv2.imread(fs[idx[0]])
    W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
    return fs, gt, k, W, H, max(64, int(math.ceil(max(W, H) / 32.0) * 32))


def det_uav(name):
    out = os.path.join(CACHE, f"e6_det_uav_{name}.pkl")
    if os.path.exists(out):
        return
    from ultralytics import YOLO
    model = YOLO(MODELS[name])
    d = {}
    for n, (i, idx) in enumerate(D.segments()):
        for S in D.SIZES:
            g = reduced(i, idx, S)
            if g is None:
                continue
            fs, gt, k, W, H, imgsz = g
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                d[(i, idx[0], S, j)] = boxes(model.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0])
        print(name, "segment", n + 1, flush=True)
    save(d, out); print("saved", out, flush=True)


# ------------------------------------------------------------ reference trackers
class Dets:
    """Results-like view of cached boxes for the Ultralytics trackers."""

    def __init__(self, d):
        self.d = np.asarray(d, float).reshape(-1, 6)

    def __len__(self):
        return len(self.d)

    def __getitem__(self, m):
        return Dets(self.d[m])

    @property
    def conf(self):
        return self.d[:, 4]

    @property
    def cls(self):
        return self.d[:, 5]

    @property
    def xywh(self):
        return self.d[:, :4]

    @property
    def xyxy(self):
        c, s = self.d[:, :2], self.d[:, 2:4] / 2
        return np.concatenate([c - s, c + s], 1)


class RefTracker:
    """Single-target view of a multi-object reference tracker (series of the matched detection)."""

    def __init__(self, kind, start):
        import ultralytics
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker
        cfg = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", f"{kind}.yaml")
        args = SimpleNamespace(**yaml.safe_load(open(cfg)))
        self.t = BYTETracker(args) if kind == "bytetrack" else BOTSORT(args)
        self.hi, self.lo = args.track_high_thresh, args.track_low_thresh
        self.start = np.array(start, float); self.tid = None
        self.series, self.pos = [], []

    def step(self, fr, truth_xy, dets):
        D_ = Dets(dets)
        out = self.t.update(D_, fr)
        hi = np.where(D_.conf >= self.hi)[0]
        lo = np.where((D_.conf > self.lo) & (D_.conf < self.hi))[0]
        matched = {}
        for o in out:
            tid, score, idx = int(o[4]), float(o[5]), int(o[7])
            pool = hi if score >= self.hi else lo
            if idx < len(pool):
                matched[tid] = D_.d[pool[idx]]
        if self.tid is None and matched:
            tid, dd = min(matched.items(), key=lambda kv: np.hypot(kv[1][0] - truth_xy[0], kv[1][1] - truth_xy[1]))
            if np.hypot(dd[0] - truth_xy[0], dd[1] - truth_xy[1]) < X.GATE_DET:
                self.tid = tid
        dd = matched.get(self.tid) if self.tid is not None else None
        if dd is not None:
            self.series.append((dd[2], dd[3])); self.pos.append((dd[0], dd[1]))
        else:
            self.series.append((np.nan, np.nan)); self.pos.append(None)


def holders(start, period):
    return [("tbd", X.TBD(start)), ("bytetrack", RefTracker("bytetrack", start)),
            ("botsort", RefTracker("botsort", start)), ("att_self", X.Holder(start, "self"))]


def step_all(hs, fr, txy, dets, on):
    for m, h in hs:
        if m == "tbd":
            h.step(dets)
        elif m == "att_self":
            h.step(fr, txy, None, dets)
        else:
            h.step(fr, txy, dets)
        p = h.pos[-1]
        on.setdefault(m, []).append(p is not None and np.hypot(p[0] - txy[0], p[1] - txy[1]) < X.ON_TARGET)


# --------------------------------------------------------------------- evaluation
def longest(s):
    ok = ~np.isnan(np.array(s, float)[:, 0]); b = c = 0
    for v in ok:
        c = c + 1 if v else 0; b = max(b, c)
    return b


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def eval_birds(series):
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    dets = {(n, r): load(os.path.join(CACHE, f"e6_det_bird_{n}_{r}.pkl")) for n in MODELS for r in BIRD_RES}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    state, on, conf = {}, {}, {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            txy = truth[tid][k]
            for key, dd in dets.items():
                d = dd.get(k, [])
                if (tid, key) not in state:
                    state[(tid, key)] = holders(txy, int(X.FPS_MIDWAY))
                step_all(state[(tid, key)], fr, txy, d, on.setdefault((tid, key), {}))
                conf.setdefault((tid, key), []).append(
                    max((b[4] for b in d if np.hypot(b[0] - txy[0], b[1] - txy[1]) < X.GATE_DET), default=0.0))
        if k % 500 == 0:
            print("birds frame", k, flush=True)
    rows = []
    for (tid, key), hs in state.items():
        for m, h in hs:
            series[("bird", tid, key, m)] = (list(h.series), on[(tid, key)][m])
            f, c = RR.decide(h.series, on[(tid, key)][m], RR.FPS_BIRD, "full")
            rows.append(dict(cls="bird", unit=tid, model=key[0], res=key[1], mode=m,
                             cov025=float(np.mean(np.array(conf[(tid, key)]) >= 0.25)),
                             on_target=float(np.mean(on[(tid, key)][m])), longest=longest(h.series),
                             decided=f, correct=c))
    return rows


def eval_uav(series):
    dets = {n: load(os.path.join(CACHE, f"e6_det_uav_{n}.pkl")) for n in MODELS}
    rows = []
    for n_, (i, idx) in enumerate(D.segments()):
        for S in D.SIZES:
            g = reduced(i, idx, S)
            if g is None:
                continue
            fs, gt, k, W, H, _ = g
            c0 = tuple(np.array([gt[idx[0]][0] + gt[idx[0]][2] / 2, gt[idx[0]][1] + gt[idx[0]][3] / 2]) / k)
            st = {n: holders(c0, int(D.FPS)) for n in MODELS}
            on = {n: {} for n in MODELS}
            cv = {n: [] for n in MODELS}
            for j in idx:
                fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
                txy = ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k)
                for n in MODELS:
                    d = dets[n][(i, idx[0], S, j)]
                    step_all(st[n], fr, txy, d, on[n])
                    cv[n].append(max((b[4] for b in d if np.hypot(b[0] - txy[0], b[1] - txy[1]) < X.GATE_DET), default=0.0))
            for n in MODELS:
                for m, h in st[n]:
                    series[("uav", (i, idx[0], S), n, m)] = (list(h.series), on[n][m])
                    f, _ = RR.decide(h.series, on[n][m], RR.FPS_UAV, "full")
                    rows.append(dict(cls="uav", unit=f"{i}:{idx[0]}", model=n, res=S, mode=m,
                                     cov025=float(np.mean(np.array(cv[n]) >= 0.25)),
                                     on_target=float(np.mean(on[n][m])), longest=longest(h.series),
                                     decided=f, correct=False))
        print("uav segment", n_ + 1, flush=True)
    return rows


def classifier_free_rows():
    """att_none / att_ext1Hz of the main run (they do not use the classifier)."""
    rows = []
    B = load(os.path.join(CACHE, "e3_series.pkl"))
    for (tid, m, sd), (s, o) in B.items():
        if m in ("att_none", "att_ext1Hz"):
            f, c = RR.decide(s, o, RR.FPS_BIRD, "full")
            rows.append(dict(cls="bird", unit=tid, model="-", res="-", mode=m, cov025=float("nan"),
                             on_target=float(np.mean(o)), longest=longest(s), decided=f, correct=c))
    U = load(os.path.join(CACHE, "e5_series.pkl"))
    for key, v in U.items():
        for (m, sd), (s, o) in v.items():
            if m in ("att_none", "att_ext1Hz"):
                f, _ = RR.decide(s, o, RR.FPS_UAV, "full")
                rows.append(dict(cls="uav", unit=f"{key[0]}:{key[1]}", model="-", res=key[2], mode=m,
                                 cov025=float("nan"), on_target=float(np.mean(o)), longest=longest(s),
                                 decided=f, correct=False))
    return rows


def summary(rows):
    L = ["E6: holding with different classifiers. 'ryad' = share of units with a gap-free held run >= 48 frames "
         "(95 % Wilson); 'on' = frames on target; birds: decision on target (primary rule); UAV: false 'bird'.", ""]
    L.append(f"{'class':5s} {'model':5s} {'res':>5s} {'mode':11s} {'n':>4s} {'cov@.25':>7s} {'on, %':>6s} "
             f"{'ryad, % (CI)':>18s} {'dec/false':>9s}")
    keys = sorted({(r["cls"], r["model"], str(r["res"]), r["mode"]) for r in rows},
                  key=lambda t: (t[0], t[2].zfill(5), t[1], t[3]))
    for c, m, res, mode in keys:
        rs = [r for r in rows if (r["cls"], r["model"], str(r["res"]), r["mode"]) == (c, m, res, mode)]
        n = len(rs); k = sum(r["longest"] >= WIN for r in rs); lo, hi = wilson(k, n)
        cov = np.nanmean([r["cov025"] for r in rs]) if not all(math.isnan(r["cov025"]) for r in rs) else float("nan")
        dec = (sum(r["correct"] for r in rs) if c == "bird" else sum(r["decided"] for r in rs))
        L.append(f"{c:5s} {m:5s} {res:>5s} {mode:11s} {n:4d} {100 * cov:6.0f}% {100 * np.mean([r['on_target'] for r in rs]):5.0f}% "
                 f"{100 * k / n:5.0f} ({lo:3.0f}-{hi:3.0f})     {dec:>4d}/{n}")
    return "\n".join(L)


def main():
    if sys.argv[1:] == ["det"]:
        for n in MODELS:
            det_uav(n)
            for r in BIRD_RES:
                det_birds(n, r)
        return
    series = {}
    rows = eval_birds(series) + eval_uav(series) + classifier_free_rows()
    save(series, os.path.join(CACHE, "e6_series.pkl"))
    with open(os.path.join(REPORTS, "e6_multidet.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text = summary(rows)
    print(text)
    open(os.path.join(REPORTS, "e6_summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

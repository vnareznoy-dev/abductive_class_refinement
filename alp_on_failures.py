#!/usr/bin/env python3
"""What the detector produces on real footage outside its training
distribution, and what the abductive core does with it.

Scene: "Kormoran (Tiefflug).webm" (Rhetos, CC BY-SA 4.0, Wikimedia Commons),
frames SEG = 1240..1439 -- a water scene with exactly one bird (the verified
track of flap_real.py, frames 1351-1398) and no drone. Every detector track
in this segment that is not the bird is therefore a false detection, and the
only correct verdict for it is `unknown`.

Reports (reports/alp_on_failures.txt):
  1. detections per frame and the share labelled `drone` at several
     confidence thresholds (the det_trust threshold on out-of-distribution data);
  2. multi-track gated association of the detections (conf 0.25): candidate
     tracks and tracks that survive the temporal-persistence requirement;
  3. for every surviving track: which angular facts pass the groundability
     gates (pipeline.angular_features) and the ALP verdict;
  4. positive control: the verified bird track with f_flap(true) from the
     silhouette channel, with and without a confident wrong detector label.

Surviving false tracks >= 48 frames are saved to reports/decoy_tracks.pkl as
(cx, cy, w, h) series; compare_sota.py uses them as protocol P4.
"""
import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flap_real as FR                                   # noqa: E402
from pipeline import Abducer, angular_features          # noqa: E402

WEIGHTS = os.environ.get("WEIGHTS", os.path.join(HERE, "rf/runs/bird_drone_clean/weights/best.pt"))
VIDEO = os.path.join(HERE, "vid2/Kormoran_Tiefflug.webm")
SEG = (1240, 1440)
IMGSZ = 1280
CONF = 0.25
THRESHOLDS = (0.10, 0.25, 0.50, 0.60)
GATE, COAST, MIN_TRACK, WIN = 60.0, 6, 25, 48
BIRD_PX = 30.0
FPS = 25.0
HFOV = 70.0
NAMES = ["bird", "drone"]


def read_segment():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, SEG[0])
    out = []
    for _ in range(SEG[1] - SEG[0]):
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


class Trk:
    def __init__(self, k, d):
        self.pts = {k: d}
        self.last, self.vel, self.miss, self.alive = d, np.zeros(2), 0, True

    def pred(self):
        return np.array(self.last[:2]) + self.vel


def associate(dets_per_frame):
    tracks = []
    for k, dets in enumerate(dets_per_frame):
        taken = set()
        for t in tracks:
            if not t.alive:
                continue
            p = t.pred()
            best, bj = GATE, None
            for j, d in enumerate(dets):
                if j in taken:
                    continue
                v = float(np.hypot(d[0] - p[0], d[1] - p[1]))
                if v < best:
                    best, bj = v, j
            if bj is None:
                t.miss += 1
                if t.miss > COAST:
                    t.alive = False
            else:
                taken.add(bj)
                d = dets[bj]
                t.vel = 0.5 * t.vel + 0.5 * (np.array(d[:2]) - np.array(t.last[:2]))
                t.last, t.miss = d, 0
                t.pts[k] = d
        for j, d in enumerate(dets):
            if j not in taken:
                tracks.append(Trk(k, d))
    return tracks


def main():
    frames = read_segment()
    H, W = frames[0].shape[:2]
    model = YOLO(WEIGHTS)
    out = [f"weights: {WEIGHTS}", f"scene: {os.path.basename(VIDEO)}, frames {SEG[0]}..{SEG[0] + len(frames) - 1} "
           f"({len(frames)} frames, {W}x{H}); truth: one bird, no drone", ""]

    raw = []                                               # (cx, cy, w, h, conf, cls) per frame, conf >= 0.10
    for fr in frames:
        r = model.predict(fr, conf=min(THRESHOLDS), imgsz=IMGSZ, device="cpu", verbose=False)[0]
        raw.append([((b.xyxy[0][0] + b.xyxy[0][2]).item() / 2, (b.xyxy[0][1] + b.xyxy[0][3]).item() / 2,
                     (b.xyxy[0][2] - b.xyxy[0][0]).item(), (b.xyxy[0][3] - b.xyxy[0][1]).item(),
                     float(b.conf[0]), int(b.cls[0])) for b in r.boxes])

    out.append(f"1. detections per frame (imgsz {IMGSZ})")
    out.append(f"{'conf >=':>8s} | {'per frame':>9s} | {'of them drone':>13s}")
    for th in THRESHOLDS:
        n = [sum(1 for d in f if d[4] >= th) for f in raw]
        nd = [sum(1 for d in f if d[4] >= th and d[5] == 1) for f in raw]
        out.append(f"{th:8.2f} | {np.mean(n):9.1f} | {np.mean(nd):13.1f}")
    out.append("")

    dets = [[d for d in f if d[4] >= CONF] for f in raw]
    tracks = associate(dets)
    total = sum(len(f) for f in dets)
    surv = [t for t in tracks if len(t.pts) >= MIN_TRACK]
    out.append(f"2. association at conf {CONF}: {total} detections on {len(frames)} frames "
               f"({total / len(frames):.1f}/frame) -> {len(tracks)} candidate tracks -> "
               f"{len(surv)} tracks >= {MIN_TRACK} frames")
    out.append("")

    # the verified bird track (frames 1351-1398) as positions; a detector track is
    # the bird if one of its points lies near the verified bird track (see below)
    bird = {int(p[0]) - SEG[0]: (p[1], p[2]) for p in FR.runs(FR.track(VIDEO, SEG[0], SEG[1])[0])[0]}
    abd = Abducer(os.path.join(HERE, "air_objects.pl"))
    out.append("3. ALP on surviving tracks (angular facts after groundability gates)")
    decoys = []
    for i, t in enumerate(surv):
        ks = sorted(t.pts)
        # nearest verified bird frame within 10 frames; the gate widens by 3 px/frame of gap
        is_bird = False
        for k in ks:
            kb = min(bird, key=lambda b: abs(b - k))
            if abs(kb - k) <= 10 and np.hypot(t.pts[k][0] - bird[kb][0],
                                              t.pts[k][1] - bird[kb][1]) < BIRD_PX + 3 * abs(kb - k):
                is_bird = True
                break
        series = [t.pts.get(k) for k in range(ks[0], ks[-1] + 1)]
        det_like = [None if d is None else (d[0], d[1], d[2]) for d in series]
        f = angular_features(det_like, W, H, HFOV, 1.0 / FPS)
        cls_votes = Counter(d[5] for d in t.pts.values())
        dc = cls_votes.most_common(1)[0][0]
        dconf = float(np.mean([d[4] for d in t.pts.values() if d[5] == dc]))
        if f is None:
            verdict, why, grounded = "unknown", "fewer than 20 points", []
        else:
            grounded = [k for k in ("f_hover", "f_turn_irreg", "f_turn_bias", "f_straight", "f_speed_cv")
                        if f.get(k) is not None and not (k == "f_hover" and f[k] == 0.0)]
            verdict, why = abd.explain(f, NAMES[dc], dconf)
        tag = "BIRD TRACK" if is_bird else "false track"
        out.append(f"   track {i}: frames {ks[0] + SEG[0]}-{ks[-1] + SEG[0]} ({len(ks)} pts), {tag}; "
                   f"detector says {NAMES[dc]} ({dconf:.2f}); grounded facts: {grounded or 'none'}; "
                   f"verdict: {verdict}  [{why}]")
        if not is_bird and len(ks) >= WIN:
            decoys.append((f"decoy:{i}", FPS, [(d[0], d[1], d[2], d[3]) for d in
                                                (t.pts[k] for k in ks)]))
    out.append("")

    ctl = {"f_flap": True}
    for label, dcls, dconf in (("no detector label", None, 0.0), ("detector says drone 0.95 (injected)", "drone", 0.95)):
        v, why = abd.explain(ctl, dcls, dconf)
        out.append(f"4. positive control, bird track, {label}: {v}  [{why}]")

    with open(os.path.join(HERE, "reports/decoy_tracks.pkl"), "wb") as fh:
        pickle.dump(decoys, fh)
    out.append(f"\nsaved {len(decoys)} false tracks >= {WIN} frames to reports/decoy_tracks.pkl (protocol P4)")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "reports/alp_on_failures.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())

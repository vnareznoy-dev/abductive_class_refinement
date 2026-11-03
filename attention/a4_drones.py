#!/usr/bin/env python3
"""E5. Holding a small UAV: DUT Anti-UAV tracking set (Zhao et al., IEEE T-ITS 2022),
20 RGB sequences with per-frame ground-truth boxes (1920x1080 / 1280x720).

Equivalent range. The UAVs in DUT are 34-158 px; a UAV of S px is obtained by
reducing the frame by k = S0 / S (S0 = median box size of the segment). This
models the same scene at a k times larger distance for a camera of k times
lower resolution (the range model of the Article 3 repository); atmosphere is
not modelled. Method parameters are NOT rescaled (window 120 px, sigma 7,
gates 30/45 px), and the detector runs at the native resolution of the reduced
frame (no further reduction) -- both choices favour the baselines, not the
proposed method.

Pre-registered segment rule: frames where the UAV is darker than its local
background by >= 8 grey levels and the background ring is uniform (std <= 25),
contiguous runs of >= 96 frames; segments longer than 300 frames are split.

Outputs: reports/e5_drones.csv, reports/e5_summary.txt
"""
import csv
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, REPORTS, WEIGHTS, WIN, band_test, local_blobs   # noqa: E402
import a4_experiments as X                                             # noqa: E402
from ultralytics import YOLO                                            # noqa: E402

DUT = os.path.join(REPO, "dut")
SIZES = (6, 9, 12)
SEG_MIN, SEG_MAX = 96, 300
DARK, RING_STD = -8.0, 25.0
GATE_DET, ON_T = 45.0, 25.0
FPS = 25.0
SEEDS = (7, 11, 23, 42, 77)
SERIES = {}


def load_seq(i):
    d = os.path.join(DUT, f"img/Anti-UAV-Tracking-V0/video{i:02d}")
    fs = sorted(glob.glob(d + "/*.jpg"))
    gt = [list(map(float, l.replace(",", " ").split()[:4]))
          for l in open(os.path.join(DUT, f"gt/Anti-UAV-Tracking-V0GT/video{i:02d}_gt.txt")) if l.strip()]
    return fs, gt


def sky_frame(img_gray, box):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return False
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
    obj = img_gray[y0:y1, x0:x1]
    p = int(max(w, h))
    ring = img_gray[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p].astype(float).copy()
    ring[min(p, y0):min(p, y0) + (y1 - y0), min(p, x0):min(p, x0) + (x1 - x0)] = np.nan
    if obj.size == 0 or np.isnan(ring).all():
        return False
    return (float(np.percentile(obj, 20)) - float(np.nanmedian(ring)) <= DARK) and float(np.nanstd(ring)) <= RING_STD


def segments():
    out = []
    for i in range(1, 21):
        fs, gt = load_seq(i)
        ok = []
        for j, (f, b) in enumerate(zip(fs, gt)):
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            ok.append(sky_frame(g, b))
        run = []
        for j, v in enumerate(ok + [False]):
            if v:
                run.append(j)
            else:
                if len(run) >= SEG_MIN:
                    for s in range(0, len(run), SEG_MAX):
                        part = run[s:s + SEG_MAX]
                        if len(part) >= SEG_MIN:
                            out.append((i, part))
                run = []
    return out


def run_segment(model, i, idx, S):
    fs, gt = load_seq(i)
    s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx]))
    k = s0 / S
    if k < 1.0:
        return None
    first = cv2.imread(fs[idx[0]])
    W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
    imgsz = max(64, int(math.ceil(max(W, H) / 32.0) * 32))
    c0 = np.array([gt[idx[0]][0] + gt[idx[0]][2] / 2, gt[idx[0]][1] + gt[idx[0]][3] / 2]) / k
    holders = [("tbd", None, X.TBD(tuple(c0))), ("att_none", None, X.Holder(tuple(c0), "none"))]
    for sd in SEEDS:
        holders.append(("att_ext1Hz", sd, X.Holder(tuple(c0), "ext", period=int(FPS), sigma=15.0, seed=sd)))
    conf, vis, on = [], [], {}
    for j in idx:
        fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
        x, y, w, h = gt[j]
        tx, ty = (x + w / 2) / k, (y + h / 2) / k
        r = model.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0]
        dets = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                 float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        near = [d for d in dets if np.hypot(d[0] - tx, d[1] - ty) < GATE_DET]
        best = max(near, key=lambda d: d[4], default=None)
        conf.append((best[4], best[5]) if best else (0.0, -1))
        vis.append(any(np.hypot(b[0] - tx, b[1] - ty) < ON_T for b in local_blobs(fr, tx, ty)))
        for mode, sd, obj in holders:
            if mode == "tbd":
                obj.step(dets)
            else:
                obj.step(fr, (tx, ty), None, dets)
            ps = obj.pos[-1]
            on.setdefault((mode, sd), []).append(ps is not None and np.hypot(ps[0] - tx, ps[1] - ty) < ON_T)
    SERIES[(i, idx[0], S)] = {(m, sd): (list(o.series), list(on[(m, sd)])) for m, sd, o in holders}
    rows = []
    cs = [c for c, _ in conf]
    base = dict(seq=i, f0=idx[0], n=len(idx), size_px=S, k=round(k, 2), frame_w=W,
                cov_05=np.mean([c >= 0.5 for c in cs]), cov_025=np.mean([c >= 0.25 for c in cs]),
                cov_01=np.mean([c >= 0.1 for c in cs]),
                hole_025=X.holes([c >= 0.25 for c in cs]), run_025=X.tbd_longest([c >= 0.25 for c in cs]),
                run_01=X.tbd_longest([c >= 0.1 for c in cs]),
                label_drone=np.mean([l == 1 for c, l in conf if c >= 0.25]) if any(c >= 0.25 for c in cs) else float("nan"),
                vis=np.mean(vis), vis_hole=X.holes(vis))
    for mode, sd, obj in holders:
        o = np.array(on[(mode, sd)], bool)
        s = np.array(obj.series, float)
        held = ~np.isnan(s[:, 0])
        best = cur = 0
        for h_ in held:
            cur = cur + 1 if h_ else 0
            best = max(best, cur)
        fired = False
        for st in range(0, len(s) - WIN + 1):
            wnd = s[st:st + WIN]
            if np.isnan(wnd).any():
                continue
            if any((r := band_test(wnd[:, col], FPS)) and r["fired"] for col in (0, 1)):
                fired = True
                break
        rows.append(dict(base, mode=mode, seed=-1 if sd is None else sd, on_target=float(o.mean()),
                         longest_run=best, measurable=best >= WIN, false_bird=fired))
    return rows


def main():
    segs = segments()
    model = YOLO(WEIGHTS)
    print(f"segments on open sky: {len(segs)} ({sum(len(p) for _, p in segs)} frames) "
          f"from sequences {sorted({i for i, _ in segs})}", flush=True)
    rows = []
    for n, (i, idx) in enumerate(segs):
        for S in SIZES:
            r = run_segment(model, i, idx, S)
            if r:
                rows += r
        print(f"  segment {n + 1}/{len(segs)} done (video{i:02d}, {len(idx)} frames)", flush=True)
    from common import CACHE, save
    save(SERIES, os.path.join(CACHE, "e5_series.pkl"))
    with open(os.path.join(REPORTS, "e5_drones.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [f"E5 drones (DUT Anti-UAV): {len(segs)} open-sky segments, "
         f"{sum(len(p) for _, p in segs)} frames, sequences {sorted({i for i, _ in segs})}", ""]
    L.append(f"{'S, px':>6s} | {'cov@0.5':>7s} {'cov@0.25':>8s} {'cov@0.1':>7s} | {'win48 @0.25/@0.1':>16s} | "
             f"{'label drone':>11s} | {'window vis':>10s} | {'held on target (none/ext1)':>27s} | "
             f"{'series>=48 (tbd/none/ext1)':>27s} | {'false bird (none/ext1)':>22s}")
    for S in SIZES:
        rs = [r for r in rows if r["size_px"] == S]
        b = [r for r in rs if r["mode"] == "att_none"]
        e = [r for r in rs if r["mode"] == "att_ext1Hz"]
        t = [r for r in rs if r["mode"] == "tbd"]
        if not b:
            continue
        ld = [r["label_drone"] for r in b if not (isinstance(r["label_drone"], float) and math.isnan(r["label_drone"]))]
        L.append(f"{S:>6d} | {100 * np.median([r['cov_05'] for r in b]):6.0f}% {100 * np.median([r['cov_025'] for r in b]):7.0f}% "
                 f"{100 * np.median([r['cov_01'] for r in b]):6.0f}% | "
                 f"{sum(r['run_025'] >= WIN for r in b):>3d}/{len(b):<3d} {sum(r['run_01'] >= WIN for r in b):>3d}/{len(b):<3d}     | "
                 f"{(100 * np.mean(ld)) if ld else float('nan'):10.0f}% | {100 * np.median([r['vis'] for r in b]):9.0f}% | "
                 f"{100 * np.mean([r['on_target'] for r in b]):12.0f}% / {100 * np.mean([r['on_target'] for r in e]):3.0f}%       | "
                 f"{sum(r['measurable'] for r in t)}/{len(t)} {sum(r['measurable'] for r in b)}/{len(b)} "
                 f"{100 * np.mean([r['measurable'] for r in e]):.0f}%            | "
                 f"{sum(r['false_bird'] for r in b)}/{len(b)} {sum(r['false_bird'] for r in e)}/{len(e)}")
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "e5_summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

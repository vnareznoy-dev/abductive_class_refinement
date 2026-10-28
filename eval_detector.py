#!/usr/bin/env python3
"""Evaluation of the two-class detector on the clip-separated validation split.

Reproduces, for the weights produced by train2.py:
  1. per-class P / R / mAP50 / mAP50-95 (ultralytics val, imgsz 640);
  2. the label confusion matrix at IoU >= 0.5, conf >= 0.25 (greedy matching,
     highest confidence first), with Wilson 95 % intervals for the two label
     error directions;
  3. the confidence distribution of correct and wrong labels -- the basis of
     the det_trust finding (a single global confidence threshold removes most
     correct bird labels while keeping most wrong ones).

Unit of count: GT boxes of rf/ds3/val (685 images). Output: reports/eval_detector.txt
"""
import glob
import os
import sys

import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("WEIGHTS", os.path.join(HERE, "rf/runs/bird_drone_clean/weights/best.pt"))
DATA = os.path.join(HERE, "rf/ds3/data.yaml")
VAL = os.path.join(HERE, "rf/ds3/val")
NAMES = ["bird", "drone"]
CONF, IOU_MATCH, TRUST = 0.25, 0.5, 0.50


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - hw), min(1.0, c + hw)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_boxes(label_path, W=640, H=640):
    out = []
    if os.path.exists(label_path):
        for ln in open(label_path):
            p = ln.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                out.append((c, ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
    return out


def main():
    lines = [f"weights: {WEIGHTS}", f"validation: {VAL} ({len(os.listdir(os.path.join(VAL, 'images')))} images)", ""]
    model = YOLO(WEIGHTS)

    m = model.val(data=DATA, imgsz=640, batch=16, device="cpu", plots=False, verbose=False)
    lines.append("1. detection quality (ultralytics val)")
    lines.append(f"{'class':>6s} {'P':>6s} {'R':>6s} {'mAP50':>6s} {'mAP50-95':>9s}")
    for i, name in enumerate(NAMES):
        p, r, ap50, ap = m.box.class_result(i)
        lines.append(f"{name:>6s} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:9.3f}")
    lines.append(f"{'all':>6s} {m.box.mp:6.3f} {m.box.mr:6.3f} {m.box.map50:6.3f} {m.box.map:9.3f}")
    lines.append("")

    cm = np.zeros((2, 3), int)                    # truth x (bird, drone, missed)
    confs = {(t, p): [] for t in (0, 1) for p in (0, 1)}
    n_img = {0: 0, 1: 0}
    for im in sorted(glob.glob(os.path.join(VAL, "images", "*"))):
        lb = os.path.join(VAL, "labels", os.path.splitext(os.path.basename(im))[0] + ".txt")
        gts = gt_boxes(lb)
        for c in {g[0] for g in gts}:
            n_img[c] += 1
        res = model.predict(im, conf=CONF, imgsz=640, device="cpu", verbose=False)[0]
        preds = sorted(((float(b.conf[0]), int(b.cls[0]), tuple(map(float, b.xyxy[0]))) for b in res.boxes),
                       reverse=True)
        used = set()
        for cf, pc, pb in preds:
            best, bj = IOU_MATCH, None
            for j, (gc, gb) in enumerate(gts):
                if j in used:
                    continue
                v = iou(pb, gb)
                if v >= best:
                    best, bj = v, j
            if bj is not None:
                used.add(bj)
                tc = gts[bj][0]
                cm[tc, pc] += 1
                confs[(tc, pc)].append(cf)
        for j, (gc, _) in enumerate(gts):
            if j not in used:
                cm[gc, 2] += 1

    lines.append(f"2. label confusion (IoU >= {IOU_MATCH}, conf >= {CONF}); unit = GT box")
    lines.append(f"   images with birds: {n_img[0]}, with drones: {n_img[1]}")
    lines.append(f"{'truth':>7s} | {'bird':>6s} {'drone':>6s} {'missed':>7s}")
    for t in (0, 1):
        lines.append(f"{NAMES[t]:>7s} | {cm[t, 0]:6d} {cm[t, 1]:6d} {cm[t, 2]:7d}")
    for t, w in ((0, 1), (1, 0)):
        k, n = cm[t, w], cm[t, 0] + cm[t, 1]
        p, lo, hi = wilson(k, n)
        lines.append(f"   {NAMES[t]} -> {NAMES[w]}: {k}/{n} = {100 * p:.1f} % [{100 * lo:.1f}; {100 * hi:.1f}] (Wilson 95 %)")
    lines.append("")

    lines.append(f"3. confidence of matched labels (share >= det_trust {TRUST})")
    for (t, p), label in (((0, 0), "correct bird->bird"), ((0, 1), "WRONG bird->drone"),
                          ((1, 1), "correct drone->drone"), ((1, 0), "WRONG drone->bird")):
        v = np.array(confs[(t, p)])
        if len(v):
            lines.append(f"   {label:22s} n={len(v):5d}  median {np.median(v):.3f}  share>={TRUST}: {100 * np.mean(v >= TRUST):.0f} %")
        else:
            lines.append(f"   {label:22s} n=    0")

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports/eval_detector.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())

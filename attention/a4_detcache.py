#!/usr/bin/env python3
"""Detector boxes on every frame of the Midway clip and the Kormoran segment
(full frame, imgsz 640, conf >= 0.05, CPU). Weights: the two-class detector
retrained by train2.py of the Article 3 repository.

Output: cache/det_midway.pkl, cache/det_kormoran.pkl -> {frame: [(cx, cy, w, h, conf, cls), ...]}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MIDWAY, KORMORAN, WEIGHTS, CACHE, frames, save  # noqa: E402
from ultralytics import YOLO                                        # noqa: E402

IMGSZ, CONF = 640, 0.05


def run(path, out, f0=0, f1=None):
    model = YOLO(WEIGHTS)
    res = {}
    for k, fr in frames(path, f0, f1):
        r = model.predict(fr, conf=CONF, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        res[k] = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                   float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                   float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        if k % 200 == 0:
            print(os.path.basename(path), k, flush=True)
    save(res, os.path.join(CACHE, out))
    print("saved", out, len(res), "frames")


if __name__ == "__main__":
    run(KORMORAN, "det_kormoran.pkl", 1240, 1440)
    run(MIDWAY, "det_midway.pkl")

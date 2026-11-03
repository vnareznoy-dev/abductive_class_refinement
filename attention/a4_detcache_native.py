#!/usr/bin/env python3
"""E1b: detector at NATIVE resolution (imgsz 1920) on the Midway clip, to check
that the classifier gaps are not an artefact of the 3x reduction to imgsz 640.
Only frames covered by the evaluation tracks are processed.
Output: cache/det_midway_1920.pkl"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MIDWAY, WEIGHTS, CACHE, frames, load, save   # noqa: E402
from ultralytics import YOLO                                     # noqa: E402

tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
need = {int(p[0]) for t in tracks for p in t["pts"]}
model = YOLO(WEIGHTS)
res = {}
for k, fr in frames(MIDWAY, min(need), max(need)):
    if k not in need:
        continue
    r = model.predict(fr, conf=0.05, imgsz=1920, device="mps", verbose=False)[0]
    res[k] = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
               float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
               float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
    if len(res) % 200 == 0:
        print(len(res), "frames", flush=True)
save(res, os.path.join(CACHE, "det_midway_1920.pkl"))
print("saved", len(res))

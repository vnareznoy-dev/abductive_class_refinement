#!/usr/bin/env python3
"""Two-class (bird / drone) RGB detector.

Purpose in this work: the existing detector is SINGLE-class and trained on
simulator imagery, so it cannot ever emit the label `bird` -- which makes the
label-revision step of the abductive method impossible to demonstrate in
practice. This model supplies a real `f_det(Class, P)` channel with both
classes.

imgsz = 640 because EVERY image in the set is natively 640x640; training at
960 only upscales, costing 2.25x compute for no added information.

Trained on rf/ds3, the CLIP-SEPARATED split (see split_clean.py). The earlier
rf/ds2 split was random over video frames, which left 97 % of validation frames
with a training frame at +-1 -- its metrics measured memorisation, not
generalisation.
"""
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    m = YOLO(os.environ.get("YOLO_BASE", "yolov8s.pt"))
    m.train(
        data=os.path.join(HERE, "rf/ds3/data.yaml"),
        epochs=40, imgsz=640, batch=16, device="mps",
        patience=12, workers=4, seed=7, deterministic=True,
        project=os.path.join(HERE, "rf/runs"), name="bird_drone_clean",
        exist_ok=True, val=True, plots=False, verbose=True,
    )

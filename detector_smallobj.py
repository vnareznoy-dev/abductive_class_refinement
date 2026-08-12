#!/usr/bin/env python3
"""Can the trained detector HOLD the small object -- i.e. replace the blob tracker?

Why this matters. The silhouette channel's binding constraint is data association:
a contrast-based blob tracker is equally happy to lock onto a wave crest or a twig
(see the article). A detector that knows what a bird looks like would not be, and
it would supply BOTH facts from one instrument: `f_det` (the class) and the box
geometry over time (the silhouette series).

The obstacle is scale, and it is measurable. In training, bird boxes were 40-80 px
inside 640x640 images. In the Kormoran footage the bird is 11x8 px inside
1920x1080; at imgsz=640 the frame is downscaled 3x, leaving ~4 px. A detector
never shown a 4-px bird cannot be expected to find one.

So the question is not "does it work" but "at what scale does it start working",
which decides whether tiled inference is a viable route. This script sweeps the
crop window and the upscale factor around KNOWN bird positions (taken from the
verified track of flap_real.py) and reports:

  * detection rate,
  * the class assigned,
  * whether the box actually lands on the bird (IoU against the verified box).

Ground truth here is the blob track that was confirmed frame-by-frame visually,
so this measures the detector against a checked reference rather than an assumption.
"""
from __future__ import annotations

import os, sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WEIGHTS = os.path.join(HERE, "rf/runs/bird_drone_clean/weights/best.pt")
NAMES = ["bird", "drone"]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ar = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / ar if ar > 0 else 0.0


def main():
    from ultralytics import YOLO
    from flap_real import track, runs, VID

    if not os.path.exists(WEIGHTS):
        sys.exit(f"немає ваг: {WEIGHTS}")
    model = YOLO(WEIGHTS)

    # verified reference track (frame-by-frame visually confirmed, see the article)
    seq, fps = track(VID, 1240, 1800)
    rs = runs(seq)
    ref = rs[0]
    print(f"еталонний трек: кадри {ref[0][0]}..{ref[-1][0]}, {len(ref)} кадрів, "
          f"рамка {np.mean([p[3] for p in ref]):.1f}x{np.mean([p[4] for p in ref]):.1f} px")
    print("(візуально підтверджений покадрово — розд. 11.2)")

    cap = cv2.VideoCapture(VID)
    ok, fr0 = cap.read()
    H, W = fr0.shape[:2]
    print(f"кадр {W}x{H}; у навчанні птахи були 40-80 px усередині 640x640\n")

    # sample every 4th frame of the reference run
    sample = ref[::4]

    print("A. ВЕСЬ КАДР, різна роздільність інференса")
    print(f"{'imgsz':>7s} | {'знайдено':>9s} | {'на птаху':>9s} | {'клас':>16s} | "
          f"{'розмір птаха':>13s}")
    print("-" * 72)
    for imgsz in (640, 1280, 1920, 2560):
        hit = onbird = 0
        cls_cnt = {}
        for p in sample:
            k, cx, cy, bw, bh, _ = p
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, fr = cap.read()
            if not ok:
                continue
            gt = (cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2)
            r = model.predict(source=fr, verbose=False, conf=0.10,
                              imgsz=imgsz)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            hit += 1
            best, bc = 0.0, None
            for j in range(len(r.boxes)):
                bb = r.boxes.xyxy[j].cpu().numpy()
                v = iou(gt, bb)
                if v > best:
                    best, bc = v, int(r.boxes.cls[j])
            if best >= 0.1:
                onbird += 1
                cls_cnt[NAMES[bc]] = cls_cnt.get(NAMES[bc], 0) + 1
        eff = bw * imgsz / W
        top = ", ".join(f"{k}:{v}" for k, v in cls_cnt.items()) or "—"
        print(f"{imgsz:7d} | {hit:4d}/{len(sample):<4d} | {onbird:4d}/{len(sample):<4d} | "
              f"{top:>16s} | {eff:10.1f} px")

    print()
    print("B. ВІКНО НАВКОЛО ОБʼЄКТА, різне збільшення")
    print("   (моделює тайловий інференс: скільки треба збільшити, щоб детектор побачив)")
    print(f"{'вікно':>7s} {'масштаб':>8s} | {'знайдено':>9s} | {'на птаху':>9s} | "
          f"{'клас':>16s} | {'розмір птаха':>13s}")
    print("-" * 82)
    for win in (320, 200, 128):
        for out in (640,):
            scale = out / win
            hit = onbird = 0
            cls_cnt = {}
            for p in sample:
                k, cx, cy, bw, bh, _ = p
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
                ok, fr = cap.read()
                if not ok:
                    continue
                x0 = int(np.clip(cx - win/2, 0, W - win))
                y0 = int(np.clip(cy - win/2, 0, H - win))
                crop = fr[y0:y0+win, x0:x0+win]
                crop = cv2.resize(crop, (out, out),
                                  interpolation=cv2.INTER_CUBIC)
                gt = ((cx-bw/2-x0)*scale, (cy-bh/2-y0)*scale,
                      (cx+bw/2-x0)*scale, (cy+bh/2-y0)*scale)
                r = model.predict(source=crop, verbose=False, conf=0.10,
                                  imgsz=out)[0]
                if r.boxes is None or not len(r.boxes):
                    continue
                hit += 1
                best, bc = 0.0, None
                for j in range(len(r.boxes)):
                    bb = r.boxes.xyxy[j].cpu().numpy()
                    v = iou(gt, bb)
                    if v > best:
                        best, bc = v, int(r.boxes.cls[j])
                if best >= 0.1:
                    onbird += 1
                    cls_cnt[NAMES[bc]] = cls_cnt.get(NAMES[bc], 0) + 1
            top = ", ".join(f"{k}:{v}" for k, v in cls_cnt.items()) or "—"
            print(f"{win:7d} {scale:7.1f}x | {hit:4d}/{len(sample):<4d} | "
                  f"{onbird:4d}/{len(sample):<4d} | {top:>16s} | "
                  f"{bw*scale:10.1f} px")
    cap.release()

    print()
    print("трактування: колонка «на птаху» — це і є здатність УТРИМАТИ обʼєкт.")
    print("«знайдено» без «на птаху» означає, що детектор бачить щось інше —")
    print("та сама проблема асоціації, лише іншим інструментом.")


if __name__ == "__main__":
    main()

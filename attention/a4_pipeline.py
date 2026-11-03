#!/usr/bin/env python3
"""Proposed coupling of the classifier and the attention method (end-to-end, no ground truth).

    classifier (full frame)  --finds the target-->  attention window (Holder)
         ^                                               |
         |  ROI crop of the window, upsampled            |  held position x_t,
         +-----------------------------------------------+  silhouette w_t, h_t
    classifier (ROI)  --cue (x, y) after a loss-->  window
                      --class votes along the held track-->  track label
    spectral criterion (6) on the longest held run   --> 'bird' evidence

Final label of the target (fixed BEFORE testing):
  'птах'      -- criterion (6) fired on the longest held run (>= 48 frames) and the
                 classifier votes do not say 'drone' (see conflict);
  'БпЛА'      -- criterion did not fire and >= MIN_VOTES classifier votes on the held
                 track, of them >= DRONE_SHARE labelled 'drone';
  'невідомо'  -- otherwise, including the conflict 'criterion fired, votes say drone'.

Only the window holds the target; the classifier never extends the track by itself.
Nothing in this file uses the true position of the target.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WIN, band_test   # noqa: E402
import a4_experiments as X          # noqa: E402

ROI, ZOOM = 120, 640            # window side in the frame, classifier input side for the crop
CONF_START, CONF_VOTE = 0.5, 0.25
MIN_VOTES, DRONE_SHARE = 5, 0.7
CLS = {0: "bird", 1: "drone"}   # classes of the Article 3 detector (rf/runs/bird_drone_clean)


def boxes(result, scale=1.0, dx=0.0, dy=0.0):
    """Ultralytics result -> [(cx, cy, w, h, conf, cls)] in full-frame pixels."""
    out = []
    for b in result.boxes:
        x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
        out.append(((x0 + x1) / 2 / scale + dx, (y0 + y1) / 2 / scale + dy,
                    (x1 - x0) / scale, (y1 - y0) / scale, float(b.conf[0]), int(b.cls[0])))
    return out


class Pipeline:
    """One target: started by the classifier, held by the window, labelled by votes + criterion."""

    def __init__(self, model, fps, imgsz_full=640, roi_zoom=True):
        self.model, self.fps, self.imgsz_full, self.roi_zoom = model, fps, imgsz_full, roi_zoom
        self.holder = None
        self.votes = []          # (class, conf) of classifier boxes on the held target
        self.log = []

    # -- 1. the classifier finds the target ------------------------------------------
    def start(self, fr):
        dets = boxes(self.model.predict(fr, conf=0.05, imgsz=self.imgsz_full, verbose=False)[0])
        cand = [d for d in dets if d[4] >= CONF_START]
        if not cand:
            return False
        d = max(cand, key=lambda d: d[4])
        self.holder = X.Holder((d[0], d[1]), "self")        # cue = classifier flashes
        self.votes.append((d[5], d[4]))
        return True

    # -- 2. the window looks, the classifier looks into the window -------------------
    def classify_roi(self, fr, cx, cy):
        H, W = fr.shape[:2]
        x0, y0 = int(np.clip(cx - ROI / 2, 0, W - ROI)), int(np.clip(cy - ROI / 2, 0, H - ROI))
        crop = fr[y0:y0 + ROI, x0:x0 + ROI]
        s = ZOOM / ROI
        up = cv2.resize(crop, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC)
        return boxes(self.model.predict(up, conf=0.05, imgsz=ZOOM, verbose=False)[0], s, x0, y0)

    def step(self, fr):
        h = self.holder
        anchor = h.last + h.vel if h.last is not None else (
            h.lost_xy + h.vel * h.since if h.lost_xy is not None else None)
        if anchor is None:
            dets = []
        elif self.roi_zoom:
            dets = self.classify_roi(fr, *anchor)
        else:
            dets = boxes(self.model.predict(fr, conf=0.05, imgsz=self.imgsz_full, verbose=False)[0])
        h.step(fr, None, None, dets)                          # the window holds; dets = cue only
        p = h.pos[-1]
        if p is not None:                                     # votes only on the held silhouette
            near = [d for d in dets if d[4] >= CONF_VOTE and np.hypot(d[0] - p[0], d[1] - p[1]) < X.ON_TARGET]
            if near:
                d = max(near, key=lambda d: d[4])
                self.votes.append((d[5], d[4]))
        self.log.append(dict(pos=p, state=h.trace[-1]["state"], n_dets=len(dets)))

    # -- 3. decision -----------------------------------------------------------------
    def label(self):
        a = np.array(self.holder.series, float)
        ok = ~np.isnan(a[:, 0]) if len(a) else np.zeros(0, bool)
        best = cur = 0; end = -1
        for i, v in enumerate(ok):
            cur = cur + 1 if v else 0
            if cur > best:
                best, end = cur, i
        fired = False
        if best >= WIN:
            run = a[end - best + 1:end + 1]
            fired = any((r := band_test(run[:, c], self.fps)) and r["fired"] for c in (0, 1))
        n = len(self.votes)
        drone = sum(1 for c, _ in self.votes if CLS.get(c) == "drone")
        votes_drone = n >= MIN_VOTES and drone / n >= DRONE_SHARE
        if fired and not votes_drone:
            lab = "птах"
        elif not fired and votes_drone:
            lab = "БпЛА"
        else:
            lab = "невідомо"
        return dict(label=lab, criterion=fired, longest_run=best, votes=n,
                    drone_share=(drone / n if n else float("nan")))

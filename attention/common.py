"""Shared paths and primitives for the Article 4 experiments.

Data and the silhouette-channel code live in the Article 3 repository
(abductive_class_refinement); this package only adds the attention experiments.
"""
import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(HERE), "flap_real.py")):
    # published layout: this folder is attention/ inside the repository
    REPO = os.path.dirname(HERE)
    A4 = HERE
else:
    # author's workspace layout: article4/code next to abductive_class_refinement/
    A4 = os.path.dirname(HERE)
    REPO = os.path.join(os.path.dirname(A4), "abductive_class_refinement")
sys.path.insert(0, REPO)
import flap_real as FR                                   # noqa: E402

MIDWAY = os.path.join(REPO, "vid3/midway.webm")
KORMORAN = os.path.join(REPO, "vid2/Kormoran_Tiefflug.webm")
WEIGHTS = os.path.join(REPO, "rf/runs/bird_drone_clean/weights/best.pt")
MIDWAY_TRACKS = os.path.join(REPO, "reports/midway_tracks.pkl")
REPORTS = os.path.join(A4, "reports")
CACHE = os.path.join(A4, "cache")
FPS_MIDWAY = 29.97
WIN = 48

band_test = FR.band_test


def frames(path, f0=0, f1=None):
    """Sequential decoding (VP9 seeks land on keyframes, so never seek per frame)."""
    cap = cv2.VideoCapture(path)
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok or (f1 is not None and k > f1):
            break
        if k >= f0:
            yield k, fr
        k += 1
    cap.release()


def local_blobs(fr, cx, cy, win=60, pct=99.0, floor=8.0):
    """Negative-contrast blobs with the threshold taken from the ATTENTION WINDOW
    (percentile of the window, not of the frame)."""
    H, W = fr.shape[:2]
    x0, x1 = max(0, int(cx - win)), min(W, int(cx + win))
    y0, y1 = max(0, int(cy - win)), min(H, int(cy + win))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    dark = cv2.GaussianBlur(g, (0, 0), 7) - g
    thr = max(float(np.percentile(dark, pct)), floor)
    m = (dark > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 4 < bw * bh < 2500 and 2 <= bw <= 70 and 2 <= bh <= 50:
            peak = float(dark[by:by + bh, bx:bx + bw].max())
            out.append((x0 + bx + bw / 2.0, y0 + by + bh / 2.0, float(bw), float(bh), peak))
    return out


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)

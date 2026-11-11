#!/usr/bin/env python3
"""Article 3, version v1.1: processing time per frame (silhouette extraction on the full frame), per spectral check and per
abductive inference (air_objects.pl via pipeline.Abducer). One thread, laptop; needs the Kormoran video and CTU_MRS_DIR (see README). Output: reports/r6_timing.txt"""
import os
import sys
import time

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import flap_real as FR                                   # noqa: E402
from pipeline import Abducer                              # noqa: E402
import evaluate as E                                      # noqa: E402
from rev_flap import fires                             # noqa: E402

cv2.setNumThreads(1)
cap = cv2.VideoCapture(FR.VID); cap.set(cv2.CAP_PROP_POS_FRAMES, 1351)
frames = [cap.read()[1] for _ in range(48)]
h, w = frames[0].shape[:2]
t0 = time.perf_counter()
for _ in range(3):
    for fr in frames:
        FR.dark_blobs(fr)
t_blob = (time.perf_counter() - t0) / (3 * len(frames))
ws, hs = np.random.default_rng(0).uniform(8, 14, 48), np.random.default_rng(1).uniform(6, 10, 48)
t0 = time.perf_counter()
for _ in range(2000):
    fires(ws, hs, 25.0)
t_spec = (time.perf_counter() - t0) / 2000
abd = Abducer(os.path.join(REPO, "air_objects.pl"))
items = E.build_set()[:20]
rng = np.random.default_rng(3)
feats = [E.featurise(p, 0.3, rng) for _, p, _ in items]
t0 = time.perf_counter()
for _ in range(5):
    for f in feats:
        abd.explain(f, "drone", 0.9)
t_alp = (time.perf_counter() - t0) / (5 * len(feats))
rep = [f"frame {w}x{h}: silhouette extraction {1000*t_blob:.1f} ms/frame",
       f"spectral criterion (12), 4 measures x 48 frames: {1000*t_spec:.2f} ms per check",
       f"abductive inference (SWI-Prolog via pyswip): {1000*t_alp:.2f} ms per decision"]
print("\n".join(rep))
open(os.path.join(REPO, "reports", "r6_timing.txt"), "w").write("\n".join(rep) + "\n")

#!/usr/bin/env python3
"""E4b. Specificity of the attention chain on non-flapping objects and background.

Data: the 9 continuous single-drone runs of birdrone v10 (2738 frames, 640x640,
the same runs as flap_specificity.py / compare_sota.py of the Article 3 repo).
For every run, attention is held
  (a) on the drone, started at its labelled position (no cue afterwards), and
      with an external cue from the label (+ noise 15 px, 0.2 Hz);
  (b) from 5 random background points per run (water / beach, >= 100 px from
      the drone), no cue -- the adversarial case: the window locks onto
      whatever contrast it finds.
Any spectral decision is a false "bird".

Output: reports/e4b_specificity_drones.txt
"""
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, REPORTS, WIN                  # noqa: E402
import a4_experiments as X                             # noqa: E402

FPS = 25.0
N_BG = 5
SEED = 7


def drone_runs():
    got = {}
    for sub in ("train", "valid"):
        for im in glob.glob(os.path.join(REPO, "rf/birdrone", sub, "images", "*")):
            b = os.path.basename(im)
            stem = b.split("_jpg")[0] if "_jpg" in b else b.split(".rf.")[0]
            if not re.match(r"^\d+$", stem):
                continue
            lb = os.path.join(REPO, "rf/birdrone", sub, "labels", os.path.splitext(b)[0] + ".txt")
            if not os.path.exists(lb):
                continue
            g = [l.split() for l in open(lb) if l.strip()]
            g = [x for x in g if len(x) >= 5]
            if len(g) != 1 or int(g[0][0]) != 1:
                continue
            n = int(stem)
            _c, cx, cy, bw, bh = (float(v) for v in g[0][:5])
            got.setdefault(n, (im, cx * 640, cy * 640))
    ks = sorted(got)
    runs, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b - a == 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    return [[got[k] for k in r] for r in runs if len(r) >= WIN]


def main():
    X.FPS_MIDWAY = FPS
    rng = np.random.default_rng(SEED)
    runs = drone_runs()
    L = [f"drone runs: {len(runs)}, frames: {sum(len(r) for r in runs)}", ""]
    tot = {"drone_none": [0, 0], "drone_ext": [0, 0], "background": [0, 0]}
    for ri, run in enumerate(runs):
        _, x0, y0 = run[0]
        holders = [("drone_none", X.Holder((x0, y0), "none")),
                   ("drone_ext", X.Holder((x0, y0), "ext", period=int(5 * FPS), sigma=15.0, seed=SEED + ri))]
        for _ in range(N_BG):
            while True:
                p = rng.uniform(40, 600, 2)
                if np.hypot(p[0] - x0, p[1] - y0) >= 100:
                    break
            holders.append(("background", X.Holder(tuple(p), "none")))
        for im, cx, cy in run:
            fr = cv2.imread(im)
            for name, h in holders:
                h.step(fr, (cx, cy), None)
        cells = []
        for name, h in holders:
            fired, _, _ = X.decide(h.series, np.zeros(len(h.series), bool))
            held = np.mean([not np.isnan(s[0]) for s in h.series])
            tot[name][0] += fired; tot[name][1] += 1
            cells.append(f"{name}:{'FIRED' if fired else '-'}({100 * held:.0f}%)")
        L.append(f"run {ri + 1} ({len(run)} frames): " + "  ".join(cells))
    L.append("")
    for name, (f, n) in tot.items():
        L.append(f"false 'bird' decisions, {name}: {f} of {n} held series")
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "e4b_specificity_drones.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

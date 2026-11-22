#!/usr/bin/env python3
"""Article 3, answer to the reviewer (remark 4.2): accuracy-rejection table of the trained classifier.

Same set and seeds as rev_kin.py (100 real CTU-MRS drones + 90 synthetic birds, detector error 20 %, confidence 0.9;
the classifier is trained on noise-free features of half of the tracks and evaluated on all 190). For every grounding
error sigma the script reports:
  the proposed method: strict accuracy, number of answers "unknown", number of wrong labels;
  the trained classifier with a reject option - it answers only if max(p, 1 - p) >= tau: coverage (share of answered
  tracks), accuracy on the answered tracks, number of wrong labels, strict accuracy (a rejection counts as an error).
Run: CTU_MRS_DIR=<folder with the CTU-MRS csv files> python3 rev_reject.py   Output: reports/r7_reject.txt"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import evaluate as E                                    # noqa: E402

OUT = os.path.join(REPO, "reports", "r7_reject.txt")
SIGMAS = (0.0, 0.05, 0.15, 0.30, 0.60, 1.00)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def main():
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    abd = E.Abducer(os.path.join(REPO, "air_objects.pl"))
    items = E.build_set()
    P(f"items: {len(items)} ({sum(1 for i in items if i[0] == 'drone')} real CTU-MRS drones + "
      f"{sum(1 for i in items if i[0] == 'bird')} synthetic birds); detector error 20 %, confidence 0.9")
    cache = {}
    for sg in SIGMAS:
        rng = np.random.default_rng(3)
        cache[sg] = [(lbl, E.featurise(pos, sg, rng), tag) for lbl, pos, tag in items]
    rows = cache[SIGMAS[0]]
    idx = np.arange(len(rows)); np.random.default_rng(0).shuffle(idx)
    tr = sorted(idx[: len(idx) // 2].tolist())
    T = E.Trained()
    T.fit([E.Trained._x(rows[i][1], rows[i][0]) for i in tr], [1.0 if rows[i][0] == "drone" else 0.0 for i in tr])
    P("trained classifier: logistic regression, trained on noise-free features (sigma = 0) of 95 tracks")
    for sg in SIGMAS:
        rng = np.random.default_rng(11)
        dets, truth, fs = [], [], []
        for lbl, f, _ in cache[sg]:
            wrong = rng.random() < 0.2
            dets.append(("bird" if lbl == "drone" else "drone") if wrong else lbl); truth.append(lbl); fs.append(f)
        truth = np.array(truth); n = len(truth)
        ab = np.array([abd.explain(f, d, 0.9)[0] for f, d in zip(fs, dets)])
        P(f"sigma {sg:.2f} m: proposed method - strict accuracy {np.mean(ab == truth):.2f}, answers 'unknown' "
          f"{int(np.sum(ab == 'unknown'))} of {n} (coverage {np.mean(ab != 'unknown'):.2f}), wrong labels "
          f"{int(np.sum((ab != truth) & (ab != 'unknown')))}")
        p = np.array([1 / (1 + np.exp(-(((E.Trained._x(f, d) - T.mu) / T.sd) @ T.w))) for f, d in zip(fs, dets)])
        pred = np.where(p >= 0.5, "drone", "bird"); conf = np.maximum(p, 1 - p)
        for tau in TAUS:
            dec = conf >= tau
            acc = f"{np.mean(pred[dec] == truth[dec]):.2f}" if dec.any() else "n/a"
            P(f"   trained, tau {tau:.2f}: coverage {dec.mean():.2f}, accuracy on answered {acc}, wrong labels "
              f"{int(np.sum(dec & (pred != truth)))}, strict accuracy {np.mean(dec & (pred == truth)):.2f}")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

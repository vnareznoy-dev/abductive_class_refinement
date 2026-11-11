#!/usr/bin/env python3
"""Article 3, version v1.1: statistics of the grounding error sweep (confidence intervals, exact tests).

Re-runs the evaluate.py sweep B (100 real CTU-MRS drones + 90 synthetic birds, detector error 20 %) and the hover set
(60 synthetic hovering drones + 90 birds) with per-item predictions, and reports for every method:
  strict accuracy with Wilson 95 % CI; the trained classifier also on its 95 held-out items only (the published
  number used all 190 items, 95 of them training items); exact McNemar tests: method vs trained classifier, method vs
  detector label, method vs threshold rule (number of discordant items).
Run: CTU_MRS_DIR=<folder with the CTU-MRS csv files> python3 rev_kin.py   (see README). Output: reports/r5_kin_stats.txt"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import evaluate as E                                    # noqa: E402

OUT = os.path.join(REPO, "reports", "r5_kin_stats.txt")


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar(a, b):
    """exact two-sided McNemar p for paired correctness vectors a, b"""
    b01 = int(np.sum(a & ~b)); b10 = int(np.sum(~a & b)); n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b01, b10, min(1.0, 2 * p)


def sweep(items, sigmas, abd, seed=3):
    cache = {}
    for sg in sigmas:
        rng = np.random.default_rng(seed)
        cache[sg] = [(lbl, E.featurise(pos, sg, rng), tag) for lbl, pos, tag in items]
    rows = cache[sigmas[0]]
    idx = np.arange(len(rows)); np.random.default_rng(0).shuffle(idx)
    tr_idx = set(idx[: len(idx) // 2].tolist())
    trained = E.Trained()
    trained.fit([E.Trained._x(rows[i][1], rows[i][0]) for i in sorted(tr_idx)],
                [1.0 if rows[i][0] == "drone" else 0.0 for i in sorted(tr_idx)])
    methods = [("detector", E.m_detector), ("rule", E.m_threshold), ("trained", trained), ("abductive", E.make_abductive(abd))]
    test = np.array([i not in tr_idx for i in range(len(rows))])
    out = {}
    for sg in sigmas:
        rng = np.random.default_rng(11)
        dets, truth, fs = [], [], []
        for lbl, f, _ in cache[sg]:
            wrong = rng.random() < 0.2
            dets.append(("bird" if lbl == "drone" else "drone") if wrong else lbl); truth.append(lbl); fs.append(f)
        truth = np.array(truth)
        out[sg] = {n: np.array([fn(f, d, 0.9) for f, d in zip(fs, dets)]) == truth for n, fn in methods}
        out[sg]["_unknown"] = np.array([E.make_abductive(abd)(f, d, 0.9) == "unknown" for f, d in zip(fs, dets)])
    return out, test


def report(P, title, items, sigmas, abd):
    res, test = sweep(items, sigmas, abd)
    P(title)
    for sg in sigmas:
        r = res[sg]; n = len(r["detector"])
        cells = []
        for m in ("detector", "rule", "trained", "abductive"):
            k = int(r[m].sum()); lo, hi = wilson(k, n)
            cells.append(f"{m} {k/n:.2f} [{lo:.2f}; {hi:.2f}]")
        kt = int(r["trained"][test].sum()); nt = int(test.sum()); lo, hi = wilson(kt, nt)
        ka = int(r["abductive"][test].sum()); la, ha = wilson(ka, nt)
        P(f"  sigma {sg:.2f} m (n = {n}): " + "; ".join(cells))
        P(f"     held-out {nt} items: trained {kt/nt:.2f} [{lo:.2f}; {hi:.2f}], abductive {ka/nt:.2f} [{la:.2f}; {ha:.2f}]; abductive 'unknown' {int(r['_unknown'].sum())}")
        for other in ("trained", "detector", "rule"):
            b01, b10, p = mcnemar(r["abductive"], r[other])
            P(f"     McNemar abductive vs {other}: only abductive right {b01}, only {other} right {b10}, p = {p:.3g}")
        b01, b10, p = mcnemar(r["abductive"][test], r["trained"][test])
        P(f"     McNemar on held-out: only abductive right {b01}, only trained right {b10}, p = {p:.3g}")


def main():
    L = []
    P = lambda s="": (L.append(s), print(s, flush=True))
    abd = E.Abducer(os.path.join(REPO, "air_objects.pl"))
    items = E.build_set()
    nd = sum(1 for i in items if i[0] == "drone")
    P(f"items: {nd} real CTU-MRS drones + {len(items) - nd} synthetic birds; detector error 20 %, confidence 0.9")
    report(P, "B. general set", items, [0.0, 0.05, 0.15, 0.30, 0.60, 1.0], abd)
    items2 = E.hovering_drones(60) + [i for i in items if i[0] == "bird"]
    report(P, "C. hovering drones (60) + birds (90)", items2, [0.0, 0.30, 0.60, 1.0], abd)
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E27b (revision of 4b, reviewer 1, remark 4). 95 % cluster-bootstrap intervals for Table 2.

Columns: TBD (E7 column A), attention window (E7 column B = E27 identity=False), window with
abductive identity (E27 identity=True). Shares of correct and wrong labels; intervals of the shares
and of the differences window - TBD and identity - window. Clusters: UAV by video sequence
(unit[0], 12), birds by track (50). 2000 resamples. Output: reports/e27_table2_ci.txt"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402
from a4_assist import label_votes         # noqa: E402

NB = 2000
R = load(os.path.join(CACHE, "e10_series.pkl"))
E7 = {(r["cls"], r["unit"], int(r["res"])): r for r in csv.DictReader(open(os.path.join(REPORTS, "e7_assist.csv")))}


def rows(c, S):
    out = []   # (cluster, label TBD, label window, label identity)
    for (cc, unit), v in R.items():
        if cc != c or (S is not None and unit[2] != S):
            continue
        k = ("bird", str(unit), 640) if c == "bird" else ("drone", f"{unit[0]}:{unit[1]}", S)
        tbd = E7[k]["A_label"]
        win = label_votes(v[("att_self", None, False)][2]); idf = label_votes(v[("att_self", None, True)][2])
        assert win == E7[k]["B_label"]
        out.append((unit if c == "bird" else unit[0], tbd, win, idf))
    return out


def shares(rs, truth):
    n = len(rs)
    return np.array([[100 * sum(r[j] == truth for r in rs) / n,
                      100 * sum(r[j] not in (truth, "unknown") for r in rs) / n] for j in (1, 2, 3)])


def main():
    rng = np.random.default_rng(0)
    L = [__doc__.splitlines()[0], "  correct / wrong labels, % (95 % CI); d = difference, pp", ""]
    names = ("TBD", "window", "window+identity")
    for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
        truth = "bird" if c == "bird" else "drone"
        rs = rows(c, S); cl = {}
        for r in rs:
            cl.setdefault(r[0], []).append(r)
        keys = list(cl); B = []
        for _ in range(NB):
            pick = rng.choice(len(keys), len(keys), replace=True)
            B.append(shares([r for i in pick for r in cl[keys[i]]], truth))
        B = np.array(B); P = shares(rs, truth)
        ci = lambda a: f"{np.percentile(a, 2.5):5.1f}..{np.percentile(a, 97.5):5.1f}"
        L.append(f"{c} {S or ''}  n={len(rs)}  clusters={len(keys)}")
        for j, nm in enumerate(names):
            L.append(f"   {nm:16s} correct {P[j, 0]:5.1f} ({ci(B[:, j, 0])})  wrong {P[j, 1]:5.1f} ({ci(B[:, j, 1])})")
        for a, b, nm in ((1, 0, "window - TBD"), (2, 1, "identity - window")):
            L.append(f"   d {nm:18s} correct {P[a, 0] - P[b, 0]:+5.1f} ({ci(B[:, a, 0] - B[:, b, 0])})  "
                     f"wrong {P[a, 1] - P[b, 1]:+5.1f} ({ci(B[:, a, 1] - B[:, b, 1])})")
        L.append("")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e27_table2_ci.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

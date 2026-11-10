#!/usr/bin/env python3
"""E27 (revision of 4b, reviewer 2, remark 3). Table 2 with the column "attention window WITH
abductive identity". The SC votes of the published E10 run (cache/e10_series.pkl: classifier boxes
with conf >= 0.25 within 25 px of the held position, the same rule as E7 variant B) are labelled with
the E7 rule (>= 5 votes, >= 70 %). The identity=False labels must reproduce E7 column B
(reports/e7_assist.csv). Output: reports/e27_table2.txt"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402
from a4_assist import label_votes         # noqa: E402

R = load(os.path.join(CACHE, "e10_series.pkl"))
E7 = list(csv.DictReader(open(os.path.join(REPORTS, "e7_assist.csv"))))
e7 = {(r["cls"], r["unit"] if r["cls"] == "drone" else int(r["unit"]), int(r["res"])): r["B_label"] for r in E7}
L = ["E27: classifier labels along the SC track, window without / with abductive identity (correct / wrong / unknown, %)", ""]
chk = []
for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
    truth = "bird" if c == "bird" else "drone"
    for idf in (False, True):
        labs = []
        for (cc, unit), v in R.items():
            if cc != c or (S is not None and unit[2] != S):
                continue
            lab = label_votes(v[("att_self", None, idf)][2]); labs.append(lab)
            if not idf:
                k = ("bird", unit, 640) if c == "bird" else ("drone", f"{unit[0]}:{unit[1]}", S)
                chk.append(e7.get(k) == lab)
        n = len(labs); cor = labs.count(truth); unk = labs.count("unknown"); wr = n - cor - unk
        L.append(f"{c} {S or '':>2} {'identity' if idf else 'window  '}  n={n:2d}  correct {100*cor/n:5.1f} ({cor:2d})  "
                 f"wrong {100*wr/n:5.1f} ({wr:2d})  unknown {100*unk/n:5.1f} ({unk:2d})")
    L.append("")
L.insert(1, f"identity=False reproduces E7 column B: {sum(chk)}/{len(chk)}")
t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e27_table2.txt"), "w").write(t + "\n")

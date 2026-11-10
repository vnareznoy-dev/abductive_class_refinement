#!/usr/bin/env python3
"""E34 (revision of 4b, main variant 2 L_t). Checks of the Python twin with the coefficient 2:
  1. X.Holder(alp=True) with A4_ALP_MULT=2 reproduces the E26b runs of the variant alp_c2
     (RevHolder(gate="alp", floor=3, mult=2)) for strategies S0 and SC, all 122 units;
  2. the Prolog theory a4_track_alp.pl (thr(body_mult, 2.0)) returns the same decision as the Python twin
     on a sample of frames (the same sampling as E10: every 5th ALP frame of 20 bird tracks, every 7th
     of every UAV segment, strategy S0).
Output: reports/e34_c2_crosscheck.txt"""
import os
import sys

os.environ["A4_ALP_MULT"] = "2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                   # noqa: E402
import a4_experiments as X                           # noqa: E402
import a4_alp_eval as AE                             # noqa: E402
from common import CACHE, REPORTS, load              # noqa: E402

assert X.ALP_MULT == 2.0


def holders(start, fps):
    return [("att_none", None, True, X.Holder(start, "none", alp=True)),
            ("att_self", None, True, X.Holder(start, "self", alp=True))]


def main():
    AE.holders = holders
    R = {}
    R.update(AE.run_birds()); R.update(AE.run_uav())
    B = load(os.path.join(CACHE, "e26b_runs.pkl"))
    same = tot = 0
    for key, v in R.items():
        for (m, sd, idf), (series, on, votes) in v.items():
            ref = np.array(B[key][(m, None, "alp_c2")]["series"], float)
            tot += 1; same += ref.shape == np.array(series, float).shape and np.allclose(ref, np.array(series, float), equal_nan=True)
    msg = [f"E34: Python twin with A4_ALP_MULT=2 reproduces E26b alp_c2 series (S0, SC): {same}/{tot}",
           AE.prolog_check(AE.SAMPLE)]
    t = "\n".join(msg); print(t); open(os.path.join(REPORTS, "e34_c2_crosscheck.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

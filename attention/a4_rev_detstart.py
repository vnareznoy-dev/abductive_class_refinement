#!/usr/bin/env python3
"""E30 (revision of 4b, reviewer 1, remark 6). Start from a classifier detection vs start from the reference,
compared on the SAME units and frames (on-target tolerance r = 2L, holder with the first-version limit 1 L_t).

Reads cache/e24_dist.pkl (a4_e13_2s.py) and cache/eval_tracks.pkl. Reports, per class / UAV size:
  - share of frames on the target from the reference start over all units and over the units where a detection
    started the track, also restricted to the frames after the detection;
  - the same share for the start from the detection;
  - age check: the reference-start holder over its FIRST N frames, N = length of the detection-start run.
The main variant 2 L_t is recomputed by a4_e13_c2.py (E33, reports/e33_e13_c2.txt).
Output: reports/e30_detstart_subset.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402


def main():
    R = load(os.path.join(CACHE, "e24_dist.pkl"))
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    out = ["E30: reviewer 1 remark 6 - start from detection vs reference start on the SAME units / frames (r = 2L)"]
    for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
        keys = [k for k in R if k[0] == c and (S is None or k[1][2] == S)]
        st = [k for k in keys if R[k]["det"] is not None]
        g = lambda k: 2 * (size[k[1]] if c == "bird" else float(S))       # noqa: E731
        on = lambda a, k: np.mean(np.array(a, float) < g(k))               # noqa: E731
        all_ref = 100 * np.mean([on(R[k]["sys"]["on"], k) for k in keys])
        sub_ref = 100 * np.mean([on(R[k]["sys"]["on"], k) for k in st])
        sub_ref_late = 100 * np.mean([on(R[k]["sys"]["on"][R[k]["det_from"]:], k) for k in st])
        det = 100 * np.mean([on(R[k]["det"]["on"], k) for k in st])
        sz_all = np.median([size[k[1]] for k in keys]) if c == "bird" else S
        sz_st = np.median([size[k[1]] for k in st]) if c == "bird" else S
        out.append(f"  {c} {S or ''}: all units ref start {all_ref:.0f} % (n={len(keys)}); started subset ref start {sub_ref:.0f} %, "
                   f"ref start on the same frames after detection {sub_ref_late:.0f} %, detection start {det:.0f} % (n={len(st)}); "
                   f"median size all {sz_all} / started {sz_st}")
    out.append("  age check: reference-start holder over its FIRST N frames, N = length of the detection-start run of the same unit")
    for c, S in (("bird", None), ("uav", 9), ("uav", 12)):
        st = [k for k in R if k[0] == c and (S is None or k[1][2] == S) and R[k]["det"] is not None]
        g = lambda k: 2 * (size[k[1]] if c == "bird" else float(S))       # noqa: E731
        first = 100 * np.mean([np.mean(np.array(R[k]["sys"]["on"][:len(R[k]["det"]["on"])], float) < g(k)) for k in st])
        det = 100 * np.mean([np.mean(np.array(R[k]["det"]["on"], float) < g(k)) for k in st])
        out.append(f"    {c} {S or ''}: reference start, first N frames {first:.0f} % vs detection start {det:.0f} % (n={len(st)})")
    return out


if __name__ == "__main__":
    lines = main()
    print("\n".join(lines))
    if "--write" in sys.argv:
        open(os.path.join(REPORTS, "e30_detstart_subset.txt"), "w").write("\n".join(lines) + "\n")

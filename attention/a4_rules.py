#!/usr/bin/env python3
"""Decision rules on the held series (birds: cache/e3_series.pkl, UAVs: cache/e5_series.pkl).

PRIMARY rule, fixed before looking at the results (= the definition of the
criterion in Article 3): ONE spectral test on the longest gap-free held run
(>= 48 frames). Sensitivity rules:
  any   -- some 48-frame window (stride 1) fires (the rule of draft v1/v2);
  major -- >= 50 % of the gap-free 48-frame windows fire;
  pair  -- two adjacent non-overlapping 48-frame windows both fire.
A decision is "on the target" if >= 50 % of the frames it rests on are on target.
Output: reports/rules.txt
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, WIN, band_test, load   # noqa: E402

FPS_BIRD, FPS_UAV = 29.97, 25.0


def fires(w, fps):
    return any((r := band_test(w[:, c], fps)) and r["fired"] for c in (0, 1))


def decide(series, on, fps, rule):
    a = np.array(series, float); on = np.array(on, bool)
    ok = ~np.isnan(a[:, 0])
    if rule == "full":
        best, cur, end = 0, 0, -1
        for i, v in enumerate(ok):
            cur = cur + 1 if v else 0
            if cur > best:
                best, end = cur, i
        if best < WIN:
            return False, False
        s = end - best + 1
        f = fires(a[s:end + 1], fps)
        return f, f and on[s:end + 1].mean() >= 0.5
    wins = [s for s in range(0, len(a) - WIN + 1) if ok[s:s + WIN].all()]
    if not wins:
        return False, False
    fl = {s: fires(a[s:s + WIN], fps) for s in wins}
    if rule == "any":
        for s in wins:
            if fl[s]:
                return True, on[s:s + WIN].mean() >= 0.5
        return False, False
    if rule == "major":
        if np.mean(list(fl.values())) >= 0.5:
            fr = [s for s in wins if fl[s]]
            cov = np.zeros(len(a), bool)
            for s in fr:
                cov[s:s + WIN] = True
            return True, on[cov].mean() >= 0.5
        return False, False
    if rule == "pair":
        for s in wins:
            if fl[s] and fl.get(s + WIN, False):
                return True, on[s:s + 2 * WIN].mean() >= 0.5
        return False, False


RULES = ("full", "any", "major", "pair")


def main():
    L = ["Primary rule: full (one test on the longest gap-free held run, as in Article 3).", ""]
    B = load(os.path.join(CACHE, "e3_series.pkl"))
    modes = ("att_none", "att_self", "att_ext0.2Hz", "att_ext1Hz", "att_contrast_self", "tbd")
    L.append("BIRDS (50 Midway tracks): decided on target / on another object / no decision, %")
    L.append(f"{'mode':>18s} | " + " | ".join(f"{r:>17s}" for r in RULES))
    for m in modes:
        keys = [k for k in B if k[1] == m]
        cells = []
        for r in RULES:
            res = [decide(*B[k], FPS_BIRD, r) for k in keys]
            n = len(res); d = sum(x[0] for x in res); c = sum(x[1] for x in res)
            cells.append(f"{100 * c / n:4.0f} /{100 * (d - c) / n:3.0f} /{100 * (n - d) / n:3.0f}")
        L.append(f"{m:>18s} | " + " | ".join(f"{x:>17s}" for x in cells))
    L.append("")
    U = load(os.path.join(CACHE, "e5_series.pkl"))
    L.append("UAVs (DUT, open-sky segments): FALSE 'bird' decisions, % of held series")
    L.append(f"{'size, mode':>18s} | " + " | ".join(f"{r:>7s}" for r in RULES) + " | measurable (run>=48)")
    for S in (6, 9, 12):
        for m in ("att_none", "att_ext1Hz", "tbd"):
            items = [v[k] for key, v in U.items() if key[2] == S for k in v if k[0] == m]
            cells = []
            for r in RULES:
                res = [decide(s, o, FPS_UAV, r) for s, o in items]
                cells.append(f"{100 * np.mean([x[0] for x in res]):6.0f}%")
            meas = []
            for s, o in items:
                okk = ~np.isnan(np.array(s, float)[:, 0]); best = cur = 0
                for v in okk:
                    cur = cur + 1 if v else 0; best = max(best, cur)
                meas.append(best >= WIN)
            L.append(f"{str(S) + ' px ' + m:>18s} | " + " | ".join(cells) + f" | {100 * np.mean(meas):.0f}% (n={len(items)})")
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "rules.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

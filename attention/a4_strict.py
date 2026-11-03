#!/usr/bin/env python3
"""Strict holding measure for Tables 3-4 and Fig. 6 of draft v5.

A unit (bird track / UAV segment run) counts as held when its longest gap-free held
run has >= 48 frames AND >= 50 % of the frames of that run are on the target
(within 25 px of the true position). The loose measure of drafts v1-v4 (run >= 48
frames, wherever it lies) is reported next to it for reference.
Output: reports/v5_strict.json, reports/v5_strict.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, WIN, load   # noqa: E402
import a4_figs_v3 as V3                         # noqa: E402
import a4_multidet as M                         # noqa: E402
from a4_multidet_table import run_on            # noqa: E402

BIRD = [("tbd", "СЗВ, 640"), ("tbd1920", "СЗВ, 1920, поріг 0,1"), ("att_none", "У0"),
        ("att_contrast_self", "УК"), ("att_self", "УС"), ("att_self_oracle", "УС-О"),
        ("att_ext0.2Hz", "УЗ-0,2"), ("att_ext1Hz", "УЗ-1")]


def stats(items):
    n = len(items); k = sum(run_on(s, o) for s, o in items); kl = sum(V3.longest(s) >= WIN for s, o in items)
    lo, hi = M.wilson(k, n)
    return dict(n=n, k=k, pct=100 * k / n, ci=[lo, hi], loose=100 * kl / n,
                on=100 * float(np.mean([np.mean(o) for s, o in items])))


def main():
    B = load(os.path.join(CACHE, "e3_series.pkl")); U = load(os.path.join(CACHE, "e5_series.pkl"))
    out, L = {}, ["Strict measure: longest gap-free run >= 48 frames and >= 50 % of it on target.", ""]
    for m, lab in BIRD:
        items = V3.tbd_native(0.1) if m == "tbd1920" else [B[k] for k in B if k[1] == m]
        out[f"bird_{m}"] = d = stats(items)
        L.append(f"bird {lab:22s} n={d['n']:3d} strict {d['pct']:5.1f} ({d['ci'][0]:.0f}-{d['ci'][1]:.0f})  loose {d['loose']:5.1f}  on {d['on']:4.1f}")
    L.append("")
    for S in (6, 9, 12):
        for m, lab in (("tbd", "СЗВ"), ("att_none", "У0"), ("att_ext1Hz", "УЗ-1")):
            items = [v[k] for key, v in U.items() if key[2] == S for k in v if k[0] == m]
            out[f"uav_{S}_{m}"] = d = stats(items)
            L.append(f"uav {S:2d} px {lab:5s} n={d['n']:3d} strict {d['pct']:5.1f} ({d['ci'][0]:.0f}-{d['ci'][1]:.0f})  loose {d['loose']:5.1f}  on {d['on']:4.1f}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "v5_strict.txt"), "w").write(t + "\n")
    json.dump(out, open(os.path.join(REPORTS, "v5_strict.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

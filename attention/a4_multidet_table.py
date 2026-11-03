#!/usr/bin/env python3
"""E6 summary with the strict measure: the longest gap-free held run has >= 48 frames AND
>= 50 % of its frames are on the target (a run held on a neighbour does not count).
Reads cache/e6_series.pkl (classifier-dependent strategies) and the main-run caches
(att_none, att_ext1Hz). Coverage = median over units of the share of frames with c >= 0.25.
Output: reports/e6_table.txt, reports/e6_numbers.json"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, WIN, load   # noqa: E402
import a4_multidet as M                         # noqa: E402
import a4_rules as RR                           # noqa: E402


def run_on(s, o):
    a = np.array(s, float); o = np.array(o, bool); ok = ~np.isnan(a[:, 0])
    best = cur = 0; end = -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    return best >= WIN and o[end - best + 1:end + 1].mean() >= 0.5


def main():
    S = load(os.path.join(CACHE, "e6_series.pkl"))
    groups = {}
    for key, (s, o) in S.items():
        c, unit, cfg, mode = key
        res = cfg[1] if c == "bird" else unit[2]
        model = cfg[0] if c == "bird" else cfg
        groups.setdefault((c, res, model, mode), []).append((s, o))
    B = load(os.path.join(CACHE, "e3_series.pkl")); U = load(os.path.join(CACHE, "e5_series.pkl"))
    for (tid, m, sd), so in B.items():
        if m in ("att_none", "att_ext1Hz"):
            for r in M.BIRD_RES:
                groups.setdefault(("bird", r, "-", m), []).append(so)
    for k, v in U.items():
        for (m, sd), so in v.items():
            if m in ("att_none", "att_ext1Hz"):
                groups.setdefault(("uav", k[2], "-", m), []).append(so)
    rows = []
    import csv
    cov = {}
    for r in csv.DictReader(open(os.path.join(REPORTS, "e6_multidet.csv"))):
        if r["mode"] == "tbd" and r["model"] != "-":
            cov.setdefault((r["cls"], int(r["res"]), r["model"]), []).append(float(r["cov025"]))
    N = {}
    L = ["class res model mode n | coverage c>=0.25 (median) | run>=48 on target, % (95% CI) | frames on target, % | bird: decision on target / UAV: false bird"]
    order = ["tbd", "bytetrack", "botsort", "att_self", "att_none", "att_ext1Hz"]
    for c in ("bird", "uav"):
        for res in (M.BIRD_RES if c == "bird" else (6, 9, 12)):
            for model in ("ours", "v8n", "v8s", "-"):
                for mode in order:
                    it = groups.get((c, res, model, mode))
                    if not it:
                        continue
                    n = len(it); k = sum(run_on(s, o) for s, o in it); lo, hi = M.wilson(k, n)
                    fps = RR.FPS_BIRD if c == "bird" else RR.FPS_UAV
                    dec = [RR.decide(s, o, fps, "full") for s, o in it]
                    d = sum(x[1] for x in dec) if c == "bird" else sum(x[0] for x in dec)
                    cv = np.median(cov[(c, res, model)]) * 100 if model != "-" else float("nan")
                    on = 100 * np.mean([np.mean(o) for s, o in it])
                    N[f"{c}_{res}_{model}_{mode}"] = dict(n=n, k=k, pct=100 * k / n, ci=[lo, hi], cov=cv, on=on, dec=d)
                    L.append(f"{c:4s} {res:>5} {model:4s} {mode:11s} {n:4d} | {cv:5.0f}% | {100 * k / n:5.0f} ({lo:3.0f}-{hi:3.0f}) | {on:5.0f}% | {d}/{n}")
            L.append("")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e6_table.txt"), "w").write(t + "\n")
    json.dump(N, open(os.path.join(REPORTS, "e6_numbers.json"), "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()

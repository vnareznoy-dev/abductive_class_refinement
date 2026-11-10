#!/usr/bin/env python3
"""E26 analysis (revision of 4b). Reads cache/e26_runs.pkl (a4_rev_runs.py).

Per variant, class / size and strategy (SE-1 pools its 5 seeds):
  off     share of HELD frames off target (r = 2L)                         -- Table 1, published metric;
  strict  continuous series >= 48 frames, >= 50 % on target (r = 2L)       -- Table 1;
  onall   share of ALL frames of the track in which the track position, coast frames included,
          lies within 2L of the reference (lost = off)                     -- reviewer 1, remark 6;
  coast   share of frames in the coast state ("unseen" abduced or no silhouette);
  decisions of criterion (6) on the longest series: birds on target / about another object,
          UAV false "bird";
  SC labels from the classifier votes on the held position (Table 2 rule).
95 % cluster-bootstrap intervals (UAV by video, birds by track; 2000 resamples) for off, strict and onall
and for the difference variant - base. Residuals of the accepted on-target silhouettes (base variant,
L_t recomputed from the accepted sizes as in (2)): median, p90, p99, share > L_t.
Outputs: reports/e26_tables.txt, reports/e26_tables.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402
import a4_drones as D                     # noqa: E402
import a4_rules as RR                     # noqa: E402
from a4_assist import label_votes         # noqa: E402
from a4_multidet_table import run_on      # noqa: E402
from a4_rev_runs import EXTRA, VARIANTS   # noqa: E402

NB = 2000
STRATS = (("S0", "att_none"), ("SC", "att_self"), ("SE-1", "att_ext1Hz"))


def unit_stats(r, L, cls):
    d = np.array(r["d"], float); dc = np.array(r["dc"], float); s = np.array(r["series"], float)
    held = ~np.isnan(s[:, 0]); on = d < 2 * L
    fps = RR.FPS_BIRD if cls == "bird" else RR.FPS_UAV
    fired, dec_on = RR.decide(r["series"], list(on), fps, "full")
    return dict(held=int(held.sum()), off=int(np.sum(held & ~on)), strict=bool(run_on(r["series"], list(on))),
                n=len(d), onall=int(np.sum(dc < 2 * L)), coast=sum(st == "coast" for st in r["states"]),
                fired=bool(fired), dec_on=bool(dec_on))


def agg(us):
    held = sum(u["held"] for u in us); n = sum(u["n"] for u in us)
    return dict(off=100 * sum(u["off"] for u in us) / max(held, 1), strict=100 * np.mean([u["strict"] for u in us]),
                strict_k=sum(u["strict"] for u in us), units=len(us), onall=100 * sum(u["onall"] for u in us) / n,
                coast=100 * sum(u["coast"] for u in us) / n, fired=sum(u["fired"] for u in us),
                dec_on=sum(u["dec_on"] for u in us))


def boot(cl_units, rng, keys=("off", "strict", "onall")):
    """cl_units: dict cluster -> list of unit stats. Percentile CI of agg() under cluster resampling."""
    cl = list(cl_units); out = {k: [] for k in keys}
    for _ in range(NB):
        us = [u for c in rng.choice(cl, len(cl), replace=True) for u in cl_units[c]]
        a = agg(us)
        for k in keys:
            out[k].append(a[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def boot_diff(clA, clB, rng, keys=("off", "strict", "onall")):
    cl = list(clA); out = {k: [] for k in keys}
    for _ in range(NB):
        pick = rng.choice(cl, len(cl), replace=True)
        a = agg([u for c in pick for u in clA[c]]); b = agg([u for c in pick for u in clB[c]])
        for k in keys:
            out[k].append(a[k] - b[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def residuals_base(R, size):
    """Residuals of accepted on-target hold silhouettes, base variant, with L_t recomputed as in (2)."""
    out = {}
    for key, v in R.items():
        cls = key[0]; L = size[key[1]] if cls == "bird" else float(key[1][2])
        grp = "bird" if cls == "bird" else f"uav {key[1][2]}"
        for (m, sd, var), r in v.items():
            if var != "base":
                continue
            s = np.array(r["series"], float); d = np.array(r["d"], float)
            for res, _, idx in r["reslog"]:
                prev = [max(w, h) for w, h in s[:idx] if not np.isnan(w)][-15:]
                if len(prev) < 5 or not d[idx] < 2 * L:
                    continue
                Lt = max(float(np.median(prev)), 3.0)
                out.setdefault(grp, []).append((res, Lt))
    return out


def main():
    R = load(os.path.join(CACHE, "e26_runs.pkl")); rng = np.random.default_rng(0)
    VAR = list(VARIANTS)
    pb = os.path.join(CACHE, "e26b_runs.pkl")
    if os.path.exists(pb):
        for k, v in load(pb).items():
            R[k].update(v)
        VAR += list(EXTRA)
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    # reproduction of the published Table 1 (e23, 2L) for base / alp
    E = load(os.path.join(CACHE, "e23_dist.pkl")); same = []
    for key, v in E.items():
        for (m, sd, idf), (s, d, votes) in v.items():
            r = R[key][(m, sd, "alp" if idf else "base")]
            same.append(np.allclose(np.array(r["series"], float), np.array(s, float), equal_nan=True)
                        and np.allclose(r["d"], d, equal_nan=True))
    L = [f"E26: variants of the association rule; base/alp reproduce e23: {sum(same)}/{len(same)}", ""]
    J = {"reproduced": [sum(same), len(same)]}
    groups = [("bird", None)] + [("uav", S) for S in D.SIZES]
    for cls, S in groups:
        L.append(f"##### {cls} {S or ''}")
        for sname, m in STRATS:
            cl = {}   # variant -> cluster -> unit stats
            labs = {}
            for key, v in R.items():
                if key[0] != cls or (cls == "uav" and key[1][2] != S):
                    continue
                Lsz = size[key[1]] if cls == "bird" else float(S)
                c = key[1] if cls == "bird" else key[1][0]
                for (mm, sd, var), r in v.items():
                    if mm != m:
                        continue
                    cl.setdefault(var, {}).setdefault(c, []).append(unit_stats(r, Lsz, cls))
                    if m == "att_self":
                        labs.setdefault(var, []).append(label_votes([x[0] for x in r["votes"]]))
            for var in VAR:
                us = [u for lst in cl[var].values() for u in lst]; a = agg(us)
                ci = boot(cl[var], rng)
                dci = boot_diff(cl[var], cl["base"], rng) if var != "base" else None
                truth = "bird" if cls == "bird" else "drone"
                lab = labs.get(var)
                labtxt = "" if lab is None else (f" | labels correct {lab.count(truth):2d} wrong {len(lab) - lab.count(truth) - lab.count('unknown'):2d}"
                                                   f" unknown {lab.count('unknown'):2d}")
                dec = (f"dec on {a['dec_on']:3d} other {a['fired'] - a['dec_on']:3d}" if cls == "bird" else f"false bird {a['fired']:2d}")
                L.append(f"  {sname:4s} {var:6s} off {a['off']:5.1f} ({ci['off'][0]:4.1f}-{ci['off'][1]:4.1f})  "
                         f"strict {a['strict_k']:3d}/{a['units']:3d} = {a['strict']:5.1f} ({ci['strict'][0]:4.1f}-{ci['strict'][1]:4.1f})  "
                         f"on-all {a['onall']:5.1f} ({ci['onall'][0]:4.1f}-{ci['onall'][1]:4.1f})  coast {a['coast']:4.1f}  {dec}"
                         + ("" if dci is None else f" | d-base off {dci['off'][0]:+.1f}..{dci['off'][1]:+.1f} strict {dci['strict'][0]:+.1f}..{dci['strict'][1]:+.1f} "
                                                   f"onall {dci['onall'][0]:+.1f}..{dci['onall'][1]:+.1f}") + labtxt)
                J[f"{cls}_{S}_{sname}_{var}"] = dict(**a, ci=ci, dci=dci, labels=None if lab is None else
                                                     dict(correct=lab.count(truth), unknown=lab.count("unknown"), n=len(lab)))
            L.append("")
    L.append("Residual of the accepted on-target silhouettes to the prediction (base variant, all strategies), px")
    for grp, rows in residuals_base(R, size).items():
        r = np.array([x[0] for x in rows]); Lt = np.array([x[1] for x in rows])
        J[f"res_{grp}"] = dict(n=len(r), median=float(np.median(r)), p90=float(np.percentile(r, 90)),
                               p99=float(np.percentile(r, 99)), over=float(100 * np.mean(r > Lt)), sigma=float(np.median(r) / 1.1774))
        L.append(f"  {grp:7s} n={len(r):6d} median {np.median(r):4.2f} p90 {np.percentile(r, 90):5.2f} p99 {np.percentile(r, 99):5.2f} "
                 f"sigma {np.median(r) / 1.1774:4.2f}  > L_t {100 * np.mean(r > Lt):4.1f} %  (L_t median {np.median(Lt):4.1f})")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e26_tables.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e26_tables.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

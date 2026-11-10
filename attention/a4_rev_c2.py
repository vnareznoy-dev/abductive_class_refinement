#!/usr/bin/env python3
"""E32 (revision of 4b, decision of 09.11: the limit 2 L_t becomes the main variant).

Recomputes for the holder with the limit 2 L_t (E26b variant alp_c2) what the published tables
report for the limit L_t:
  * Table 3 / Fig. 5: classifier alone vs classifier + ALP on the SC track, T = 0.5, 0.4, 0.3, 0.25
    (same rule as E28 / a4_v6_runs.answer_system; the E26 alp records reproduce E28 exactly, 122/122);
  * Table 2: labels along the SC track (TBD = E7 column A, window = E7 column B, window + 2 L_t),
    with 95 % cluster-bootstrap intervals of the shares and of the differences;
  * explanation examples at T = 0.5.
Output: reports/e32_c2.txt"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402
import a4_drones as D                     # noqa: E402
import a4_experiments as X                # noqa: E402
import a4_v6_runs as V                    # noqa: E402
from a4_assist import label_votes         # noqa: E402
from a4_rev_stats import mcnemar          # noqa: E402

NB = 10000
TS = (0.5, 0.4, 0.3, 0.25)


def boot_mean(grp, arrs, rng, nb=NB):
    g = np.array(grp); uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}
    out = {k: [] for k in arrs}
    for _ in range(nb):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        for k, a in arrs.items():
            out[k].append(100 * np.mean(a[pick]))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}


def main():
    R = load(os.path.join(CACHE, "e26_runs.pkl"))
    for k, v in load(os.path.join(CACHE, "e26b_runs.pkl")).items():
        R[k].update(v)
    V6 = load(os.path.join(CACHE, "v6_runs.pkl"))
    E7 = {(r["cls"], r["unit"], int(r["res"])): r for r in csv.DictReader(open(os.path.join(REPORTS, "e7_assist.csv")))}
    alp = V.Alp3(); rng = np.random.default_rng(0)
    L = ["E32: main variant 2 L_t (alp_c2) -- Table 3 / Fig. 5 and Table 2", ""]
    keys_all = sorted(V6, key=str)
    for T in TS:
        list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz(f"thr(det_trust, {T})")
        L.append(f"=== Table 3 / Fig. 5, trust threshold {T}")
        for c, S in (("uav", None), ("uav", 6), ("uav", 9), ("uav", 12), ("bird", None)):
            keys = [k for k in keys_all if k[0] == c and (S is None or k[1][2] == S)]
            truth = "bird" if c == "bird" else "drone"; fps = X.FPS_MIDWAY if c == "bird" else D.FPS
            grp = [k[1][0] if c == "uav" else k[1] for k in keys]
            al = [V.answer_alone(V6[k]["tbd"]["votes"])[0] for k in keys]
            res = {"alone": al}
            for var in ("alp", "alp_c2"):
                res[var] = [V.answer_system(R[k][("att_self", None, var)], fps, alp)[0] for k in keys]
            n = len(keys); tag = f"{c} {S or 'all'}"
            arr = {}
            for name, labs in res.items():
                cor = labs.count(truth); unk = labs.count("unknown"); wr = n - cor - unk
                prec = 100 * cor / (cor + wr) if cor + wr else float("nan")
                arr[name] = (np.array([x == truth for x in labs]), np.array([x not in (truth, "unknown") for x in labs]))
                L.append(f"  {tag:8s} {name:6s} n={n:2d}  correct {cor:2d} ({100*cor/n:4.1f} %)  wrong {wr:2d} ({100*wr/n:4.1f} %)  "
                         f"unknown {unk:2d}  precision {prec:5.1f} %  coverage {100*(cor+wr)/n:4.1f} %")
            ca, wa = arr["alone"]; cs, ws = arr["alp_c2"]
            mw = mcnemar(wa, ws); mc = mcnemar(ca, cs)
            ci = boot_mean(grp, dict(d_cor=cs.astype(float) - ca, d_wr=ws.astype(float) - wa), rng, 2000)
            L.append(f"  {tag:8s} 2L_t vs alone: McNemar wrong {mw[0]}/{mw[1]} p = {mw[2]:.3f}; correct {mc[0]}/{mc[1]} p = {mc[2]:.3f}; "
                     f"diff correct {ci['d_cor'][0]:+.1f}..{ci['d_cor'][1]:+.1f} pp, wrong {ci['d_wr'][0]:+.1f}..{ci['d_wr'][1]:+.1f} pp")
        L.append("")
    # Table 2
    L.append("=== Table 2: labels along the SC track, correct / wrong % (95 % CI)")
    for c, S in (("bird", None), ("uav", 6), ("uav", 9), ("uav", 12)):
        truth = "bird" if c == "bird" else "drone"; rows = []
        for (cc, unit), v in R.items():
            if cc != c or (S is not None and unit[2] != S):
                continue
            k = ("bird", str(unit), 640) if c == "bird" else ("drone", f"{unit[0]}:{unit[1]}", S)
            labs = [E7[k]["A_label"]] + [label_votes([x[0] for x in v[("att_self", None, var)]["votes"]]) for var in ("base", "alp", "alp_c2")]
            assert labs[1] == E7[k]["B_label"]
            rows.append((unit if c == "bird" else unit[0], labs))
        grp = [r[0] for r in rows]; names = ("TBD", "window", "window+L_t", "window+2L_t")
        C = {nm: np.array([r[1][j] == truth for r in rows]) for j, nm in enumerate(names)}
        W = {nm: np.array([r[1][j] not in (truth, "unknown") for r in rows]) for j, nm in enumerate(names)}
        arrs = {f"c_{nm}": C[nm] for nm in names} | {f"w_{nm}": W[nm] for nm in names}
        arrs |= {"dc_win_tbd": C["window"].astype(float) - C["TBD"], "dw_win_tbd": W["window"].astype(float) - W["TBD"],
                 "dc_2l_win": C["window+2L_t"].astype(float) - C["window"], "dw_2l_win": W["window+2L_t"].astype(float) - W["window"]}
        ci = boot_mean(grp, arrs, rng, 2000)
        L.append(f"{c} {S or ''}  n={len(rows)}")
        for nm in names:
            L.append(f"   {nm:12s} correct {100*C[nm].mean():5.1f} ({ci['c_'+nm][0]:5.1f}..{ci['c_'+nm][1]:5.1f})  "
                     f"wrong {100*W[nm].mean():5.1f} ({ci['w_'+nm][0]:5.1f}..{ci['w_'+nm][1]:5.1f})")
        for a, b, tag in (("window", "TBD", "win_tbd"), ("window+2L_t", "window", "2l_win")):
            mc = mcnemar(C[b], C[a]); mw = mcnemar(W[b], W[a])
            L.append(f"   d {a} - {b}: correct {100*(C[a].mean()-C[b].mean()):+5.1f} ({ci['dc_'+tag][0]:+.1f}..{ci['dc_'+tag][1]:+.1f}, McNemar p {mc[2]:.3f})  "
                     f"wrong {100*(W[a].mean()-W[b].mean()):+5.1f} ({ci['dw_'+tag][0]:+.1f}..{ci['dw_'+tag][1]:+.1f}, McNemar p {mw[2]:.3f})")
        L.append("")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e32_c2.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

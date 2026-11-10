#!/usr/bin/env python3
"""E28 (revision of 4b, reviewer 1, remarks 3-4; reviewer 2, remark 2). Statistics of Table 3 / Fig. 5.

Reads cache/v6_runs.pkl (published E11 run). For the trust threshold T = 0.5, 0.4, 0.3, 0.25 and for
the classifier alone and the classifier + ALP:
  * full matrices correct / wrong / unknown per class and size;
  * precision of the decisions taken = correct / (correct + wrong), coverage = (correct + wrong) / n;
  * exact McNemar test (two-sided, binomial on the discordant units) for "wrong" and for "correct";
  * 95 % cluster-bootstrap intervals (10 000 resamples): UAV resampled by VIDEO SEQUENCE (12 videos,
    all segments and sizes of a video together), birds by track (one video only).
It also prints the full ALP justifications of two units (confirmed / revised) as explanation examples.
Outputs: reports/e28_stats.txt, reports/e28_stats.json
"""
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load   # noqa: E402
import a4_drones as D                     # noqa: E402
import a4_experiments as X                # noqa: E402
import a4_v6_runs as V                    # noqa: E402

TS = (0.5, 0.4, 0.3, 0.25)
NB = 10000


def mcnemar(a, b):
    """a, b: boolean arrays (event per unit). Exact two-sided binomial test on discordant pairs."""
    n10 = int(np.sum(a & ~b)); n01 = int(np.sum(~a & b)); n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n10, n01, p


def boot(groups, vals, rng):
    """groups: cluster id per unit; vals: dict name -> array per unit. Returns name -> (lo, hi) of the mean, %."""
    g = np.array(groups); uniq = np.unique(g); idx = {u: np.where(g == u)[0] for u in uniq}
    out = {k: [] for k in vals}
    for _ in range(NB):
        pick = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        for k, v in vals.items():
            out[k].append(100 * np.mean(v[pick]))
    return {k: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) for k, x in out.items()}


def main():
    R = load(os.path.join(CACHE, "v6_runs.pkl")); alp = V.Alp3(); rng = np.random.default_rng(0)
    J, L = {}, ["E28: Table 3 / Fig. 5 statistics (alone = classifier with tracking-by-detection, ALP = classifier + ALP)", ""]
    for T in TS:
        list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz(f"thr(det_trust, {T})")
        L.append(f"=== trust threshold {T}")
        for c, S in (("uav", None), ("uav", 6), ("uav", 9), ("uav", 12), ("bird", None)):
            keys = sorted(k for k in R if k[0] == c and (S is None or k[1][2] == S))
            truth = "bird" if c == "bird" else "drone"; fps = X.FPS_MIDWAY if c == "bird" else D.FPS
            al = [V.answer_alone(R[k]["tbd"]["votes"])[0] for k in keys]
            sy = [V.answer_system(R[k]["sys"], fps, alp)[0] for k in keys]
            grp = [k[1][0] if c == "uav" else k[1] for k in keys]
            n = len(keys); row = {}
            for name, labs in (("alone", al), ("ALP", sy)):
                cor = labs.count(truth); unk = labs.count("unknown"); wr = n - cor - unk
                prec = 100 * cor / (cor + wr) if cor + wr else float("nan")
                row[name] = dict(n=n, correct=cor, wrong=wr, unknown=unk, precision=prec, coverage=100 * (cor + wr) / n)
            ca = np.array([x == truth for x in al]); cs = np.array([x == truth for x in sy])
            wa = np.array([x not in (truth, "unknown") for x in al]); ws = np.array([x not in (truth, "unknown") for x in sy])
            mw = mcnemar(wa, ws); mc = mcnemar(ca, cs)
            ci = boot(grp, dict(cor_alone=ca, cor_alp=cs, wr_alone=wa, wr_alp=ws), rng)
            dci = boot(grp, dict(d_cor=cs.astype(float) - ca, d_wr=ws.astype(float) - wa), rng)
            row.update(mcnemar_wrong=mw, mcnemar_correct=mc, ci=ci, diff_ci=dci, clusters=len(set(grp)))
            J[f"T{T}_{c}_{S}"] = row
            tag = f"{c} {S or 'all'}"
            for name in ("alone", "ALP"):
                r = row[name]; k1, k2 = ("cor_alone", "wr_alone") if name == "alone" else ("cor_alp", "wr_alp")
                L.append(f"  {tag:8s} {name:5s} n={n:2d}  correct {r['correct']:2d} ({100*r['correct']/n:4.1f} %, CI {ci[k1][0]:4.1f}-{ci[k1][1]:4.1f})  "
                         f"wrong {r['wrong']:2d} ({100*r['wrong']/n:4.1f} %, CI {ci[k2][0]:4.1f}-{ci[k2][1]:4.1f})  unknown {r['unknown']:2d}  "
                         f"precision {r['precision']:5.1f} %  coverage {r['coverage']:4.1f} %")
            L.append(f"  {tag:8s} McNemar wrong: alone-only {mw[0]}, ALP-only {mw[1]}, p = {mw[2]:.3f}; "
                     f"correct: alone-only {mc[0]}, ALP-only {mc[1]}, p = {mc[2]:.3f}; "
                     f"diff ALP-alone correct CI {dci['d_cor'][0]:+.1f}..{dci['d_cor'][1]:+.1f} pp, wrong CI {dci['d_wr'][0]:+.1f}..{dci['d_wr'][1]:+.1f} pp "
                     f"({len(set(grp))} clusters)")
        L.append("")
    # explanation examples at the published threshold
    list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz("thr(det_trust, 0.5)")
    L.append("Explanation examples (T = 0.5): full justification term returned by classify/2")
    seen = set()
    for k in sorted(R, key=str):
        c = k[0]; fps = X.FPS_MIDWAY if c == "bird" else D.FPS
        rec = R[k]["sys"]; votes = rec["votes"]
        if votes and max(v for _, v in votes) >= V.CONF_SURE:
            continue
        det = None
        if len(votes) >= 5:
            cls = max(("bird", "drone"), key=lambda cc: sum(v == cc for v, _ in votes))
            det = (cls, float(np.median([cf for v, cf in votes if v == cls])))
        run = V.longest_run(rec["series"])
        flap = None if run is None else any((r := V.band_test(run[:, j], fps)) and r["fired"] for j in (0, 1))
        for f in ("f_det(_,_)", "f_flap(_)"):
            list(alp.pl.query(f"retractall({f})"))
        if det is not None:
            alp.pl.assertz(f"f_det({det[0]},{det[1]})")
        if flap is not None:
            alp.pl.assertz(f"f_flap({'true' if flap else 'false'})")
        r = list(alp.pl.query("classify(C, J)", maxresult=1))[0]
        kind = str(r["J"]).split("(")[0]
        if kind in ("confirmed", "revised") and (kind, c) not in seen:
            seen.add((kind, c))
            L.append(f"  unit {k}: facts f_det{det}, f_flap({flap}); run {0 if run is None else len(run)} frames -> "
                     f"class {r['C']}, justification {r['J']}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e28_stats.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e28_stats.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E14. Sensitivity of the classifier -> ALP decision (E11) to the trust threshold of the Article 3
ALP (thr(det_trust, T) in air_objects.pl, overridden by retract/assert as in the Article 3
sensitivity analysis; nothing else in the theory changes). T = 0.5 (authored default), 0.4, 0.3, 0.25.
Reads cache/v6_runs.pkl. Output: reports/e14_sens.txt, reports/e14_sens.json, fig/fig_sens.png"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_style as ST                     # noqa: E402
from a4_style import plt                  # noqa: E402
from common import A4, CACHE, REPORTS, load   # noqa: E402
import a4_drones as D                     # noqa: E402
import a4_experiments as X                # noqa: E402
import a4_v6_runs as V                    # noqa: E402

TS = (0.5, 0.4, 0.3, 0.25)


def main():
    R = load(os.path.join(CACHE, "v6_runs.pkl")); alp = V.Alp3()
    out, L = {}, ["E14: classifier -> ALP, trust threshold of the classifier label (det_trust)", ""]
    for T in TS:
        list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz(f"thr(det_trust, {T})")
        for c in ("bird", "uav"):
            keys = [k for k in R if k[0] == c]; truth = "bird" if c == "bird" else "drone"
            fps = X.FPS_MIDWAY if c == "bird" else D.FPS
            labs = [V.answer_system(R[k]["sys"], fps, alp)[0] for k in keys]
            d = dict(n=len(labs), correct=labs.count(truth), unknown=labs.count("unknown"),
                     wrong=len(labs) - labs.count(truth) - labs.count("unknown"),
                     d2b=labs.count("bird") if c == "uav" else 0)
            out[f"{c}_{T}"] = d
            L.append(f"T={T:<4} {c:4s} n={d['n']:2d} correct {d['correct']:2d} wrong {d['wrong']:2d} unknown {d['unknown']:2d}"
                     + (f"  drone->bird {d['d2b']}" if c == "uav" else ""))
    list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz("thr(det_trust, 0.5)")
    for c in ("bird", "uav"):
        keys = [k for k in R if k[0] == c]; truth = "bird" if c == "bird" else "drone"
        labs = [V.answer_alone(R[k]["tbd"]["votes"])[0] for k in keys]
        out[f"{c}_alone"] = dict(n=len(labs), correct=labs.count(truth), unknown=labs.count("unknown"),
                                 wrong=len(labs) - labs.count(truth) - labs.count("unknown"),
                                 d2b=labs.count("bird") if c == "uav" else 0)
        L.append(f"classifier alone {c}: {out[f'{c}_alone']}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e14_sens.txt"), "w").write(t + "\n")
    json.dump(out, open(os.path.join(REPORTS, "e14_sens.json"), "w"), indent=1)
    # figure: two panels, shares in % of units
    fig, axes = plt.subplots(1, 2, figsize=(ST.W_IN, 3.9), sharey=True)
    for ax, c, lab in zip(axes, ("uav", "bird"), ("а", "б")):
        n = out[f"{c}_0.5"]["n"]
        for key, col, ls, name in (("correct", ST.GREEN, "-", "правильно"), ("wrong", ST.ORANGE, "-", "хибно"),
                                   ("d2b", ST.PURPLE, "--", "БпЛА → «птах»")):
            if c == "bird" and key == "d2b":
                continue
            y = [100 * out[f"{c}_{T}"][key] / n for T in TS]
            ax.plot(TS, y, color=col, ls=ls, marker="o", ms=5, label=name)
            ax.axhline(100 * out[f"{c}_alone"][key] / n, color=col, ls=":", lw=1.0)
        ax.set_xlim(0.52, 0.23); ax.set_xticks(TS); ST.comma(ax)
        ax.set_xlabel("Поріг довіри до мітки", labelpad=4); ST.panel(ax, lab)
    axes[0].set_ylabel("Частка цілей, %", labelpad=8)
    h, l = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], color="black", ls=":", lw=1.0)); l.append("класифікатор без ALP")
    ST.top_legend(fig, h, l, 2, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(A4, "fig", "fig_sens.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

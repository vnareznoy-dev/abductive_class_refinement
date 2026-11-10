#!/usr/bin/env python3
"""Fig. 5 of 4b for the main variant 2 L_t (revision): dependence of the ALP conclusion on the label trust
threshold. Same rule and layout as a4_sens.py / a4_sens_en.py (E14), but the system answer is computed on the
SC track of the holder limited by 2 L_t (E26b alp_c2; the E26 alp records reproduce E14/E28 exactly, 122/122).
Classifier alone: tracking by detection (v6_runs tbd). Outputs: fig/fig_sens_c2.png, fig/en/fig_sens_c2_en.png,
reports/e35_sens_c2.txt"""
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
TXT = {"ua": dict(cor="правильно", wr="хибно", d2b="БпЛА → «птах»", alone="класифікатор без ALP", x="Поріг довіри до мітки",
                  y="Частка цілей, %", p=("а", "б"), out="fig_sens_c2.png"),
       "en": dict(cor="correct", wr="wrong", d2b="UAV → “bird”", alone="classifier without ALP", x="Label trust threshold",
                  y="Share of targets, %", p=("a", "b"), out=os.path.join("en", "fig_sens_c2_en.png"))}


def main():
    V6 = load(os.path.join(CACHE, "v6_runs.pkl")); B = load(os.path.join(CACHE, "e26b_runs.pkl")); alp = V.Alp3()
    out, L = {}, ["E35: classifier -> ALP on the 2 L_t track (alp_c2, SC), trust threshold", ""]
    for T in TS:
        list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz(f"thr(det_trust, {T})")
        for c in ("bird", "uav"):
            keys = [k for k in V6 if k[0] == c]; truth = "bird" if c == "bird" else "drone"
            fps = X.FPS_MIDWAY if c == "bird" else D.FPS
            labs = [V.answer_system(B[k][("att_self", None, "alp_c2")], fps, alp)[0] for k in keys]
            d = dict(n=len(labs), correct=labs.count(truth), unknown=labs.count("unknown"),
                     wrong=len(labs) - labs.count(truth) - labs.count("unknown"), d2b=labs.count("bird") if c == "uav" else 0)
            out[f"{c}_{T}"] = d
            L.append(f"T={T:<4} {c:4s} " + " ".join(f"{k} {v}" for k, v in d.items()))
    list(alp.pl.query("retractall(thr(det_trust, _))")); alp.pl.assertz("thr(det_trust, 0.5)")
    for c in ("bird", "uav"):
        keys = [k for k in V6 if k[0] == c]; truth = "bird" if c == "bird" else "drone"
        labs = [V.answer_alone(V6[k]["tbd"]["votes"])[0] for k in keys]
        out[f"{c}_alone"] = dict(n=len(labs), correct=labs.count(truth), unknown=labs.count("unknown"),
                                 wrong=len(labs) - labs.count(truth) - labs.count("unknown"), d2b=labs.count("bird") if c == "uav" else 0)
        L.append(f"classifier alone {c}: {out[f'{c}_alone']}")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e35_sens_c2.txt"), "w").write(t + "\n")
    from matplotlib.lines import Line2D
    for lang, tx in TXT.items():
        fig, axes = plt.subplots(1, 2, figsize=(ST.W_IN, 3.9), sharey=True)
        for ax, c, lab in zip(axes, ("uav", "bird"), tx["p"]):
            n = out[f"{c}_0.5"]["n"]
            for key, col, ls, name in (("correct", ST.GREEN, "-", tx["cor"]), ("wrong", ST.ORANGE, "-", tx["wr"]),
                                       ("d2b", ST.PURPLE, "--", tx["d2b"])):
                if c == "bird" and key == "d2b":
                    continue
                y = [100 * out[f"{c}_{T}"][key] / n for T in TS]
                ax.plot(TS, y, color=col, ls=ls, marker="o", ms=5, label=name)
                ax.axhline(100 * out[f"{c}_alone"][key] / n, color=col, ls=":", lw=1.0)
            ax.set_xlim(0.52, 0.23); ax.set_xticks(TS); ST.comma(ax)
            ax.set_xlabel(tx["x"], labelpad=4); ST.panel(ax, lab)
        axes[0].set_ylabel(tx["y"], labelpad=8)
        h, l = axes[0].get_legend_handles_labels()
        h.append(Line2D([], [], color="black", ls=":", lw=1.0)); l.append(tx["alone"])
        ST.top_legend(fig, h, l, 2, y=0.93)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(A4, "fig", tx["out"]), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()

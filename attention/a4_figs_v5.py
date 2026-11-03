#!/usr/bin/env python3
"""Figures of draft v5 in the shared style (a4_style.py).

fig/fig2_conf.png    -- classifier confidence along three bird tracks vs holding in the window
fig/fig_rules.png    -- decision rule trade-off (unchanged numbers, new style)
fig/fig_compare.png  -- E6: strict holding measure (run >= 48 frames, >= 50 % on target) for the
                        reference trackers on the best of three classifiers vs the attention window
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_figures as F           # noqa: E402  (data helpers of fig 2)
import a4_rules as RR            # noqa: E402
import a4_style as ST            # noqa: E402
from a4_style import plt         # noqa: E402
from common import A4, CACHE, REPORTS, load   # noqa: E402
from matplotlib.lines import Line2D           # noqa: E402
from matplotlib.patches import Patch          # noqa: E402

ST.apply()
FIG = os.path.join(A4, "fig")


def fig2():
    tids = F.pick_examples()
    res, tracks = F.series_for(tids)
    fig, axes = plt.subplots(3, 1, figsize=(ST.W_IN, 8.6), sharex=False)
    for ax, tid, lab in zip(axes, tids, "абв"):
        s = res[tid]; f0 = s[0][0]
        x = np.array([k - f0 for k, *_ in s]); c = np.array([v[1] for v in s]); vis = [v[2] for v in s]
        sz = np.median([max(p[3], p[4]) for p in tracks[tid]["pts"]])
        for xx, cc in zip(x, c):
            if cc < 0.1:
                ax.axvspan(xx, xx + 1, color="#E6E6E6", lw=0, zorder=0)
        ax.plot(x, c, color=ST.BLUE)
        for th, ls in ((0.5, "--"), (0.25, "-."), (0.1, ":")):
            ax.axhline(th, color=ST.GREY, lw=1.0, ls=ls)
        for xx, v in zip(x, vis):
            if v:
                ax.add_patch(plt.Rectangle((xx, -0.19), 1, 0.12, color=ST.GREEN, lw=0))
        ax.set_ylim(-0.22, 1.0); ax.set_xlim(0, x[-1] + 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("$c_t$", labelpad=8)
        ST.panel(ax, lab)
        ax.text(0.985, 0.93, f"силует {sz:.0f} пкс", transform=ax.transAxes, ha="right", va="top",
                bbox=dict(fc="white", ec="black", lw=0.6, pad=2))
        ST.comma(ax)
    axes[-1].set_xlabel("Номер кадру від початку треку", labelpad=6)
    ST.top_legend(fig, [Line2D([], [], color=ST.BLUE), Line2D([], [], color=ST.GREY, ls="--", lw=1),
                        Line2D([], [], color=ST.GREY, ls="-.", lw=1), Line2D([], [], color=ST.GREY, ls=":", lw=1),
                        Patch(fc="#E6E6E6"), Patch(fc=ST.GREEN)],
                  ["$c_t$", "поріг 0,5", "поріг 0,25", "поріг 0,1", "$c_t<0{,}1$", "силует у вікні уваги"], 3, y=1.0)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(FIG, "fig2_conf.png"), bbox_inches="tight")
    plt.close(fig)


def fig_rules():
    B = load(os.path.join(CACHE, "e3_series.pkl")); U = load(os.path.join(CACHE, "e5_series.pkl"))
    T = load(os.path.join(CACHE, "truth_series_birds.pkl"))
    rules = [("any", "будь-яке\nвікно"), ("pair", "два вікна\nпоспіль"), ("major", "більшість\nвікон"),
             ("full", "увесь\nряд")]
    bt, ba, uf = [], [], []
    for r, _ in rules:
        bt.append(100 * np.mean([RR.decide(s, [True] * len(s), RR.FPS_BIRD, r)[0] for s in T.values()]))
        ba.append(100 * np.mean([RR.decide(s, o, RR.FPS_BIRD, r)[1] for (tid, m, sd), (s, o) in B.items() if m == "att_none"]))
        it = [v[k] for key, v in U.items() for k in v if k[0] in ("att_none", "att_ext1Hz")]
        uf.append(100 * np.mean([RR.decide(s, o, RR.FPS_UAV, r)[0] for s, o in it]))
    fig, ax = plt.subplots(figsize=(ST.W_IN, 4.6))
    x = np.arange(len(rules)); w = 0.26
    bars = [(bt, "white", "птахи, еталонне утримання"), (ba, ST.BLUE, "птахи, утримання У0"),
            (uf, ST.ORANGE, "БпЛА, хибне рішення «птах»")]
    for j, (v, col, lab) in enumerate(bars):
        ax.bar(x + (j - 1) * w, v, w, color=col, ec="black", lw=0.6, label=lab)
        for xi, vi in zip(x + (j - 1) * w, v):
            ax.text(xi, vi + 1.2, f"{vi:.0f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in rules]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Частка рядів, %", labelpad=8); ax.set_ylim(0, 100)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rules.png"))
    plt.close(fig)


def fig_compare():
    N = json.load(open(os.path.join(REPORTS, "e6_numbers.json")))
    conds = [("bird", 640, "птахи\n640"), ("bird", 1920, "птахи\n1920"), ("uav", 6, "БпЛА\n6 пкс"),
             ("uav", 9, "БпЛА\n9 пкс"), ("uav", 12, "БпЛА\n12 пкс")]
    modes = [("tbd", "СЗВ*"), ("bytetrack", "ByteTrack*"), ("botsort", "BoT-SORT*"), ("att_self", "УС*"),
             ("att_none", "У0"), ("att_ext1Hz", "УЗ-1")]
    fig, ax = plt.subplots(figsize=(ST.W_IN, 4.8))
    x = np.arange(len(conds)); w = 0.135
    for j, (m, lab) in enumerate(modes):
        v, lo, hi = [], [], []
        for c, r, _ in conds:
            if m in ("att_none", "att_ext1Hz"):
                d = N[f"{c}_{r}_-_{m}"]
            else:   # best of the three classifiers for this strategy
                d = max((N[f"{c}_{r}_{k}_{m}"] for k in ("ours", "v8n", "v8s")), key=lambda q: q["pct"])
            v.append(d["pct"]); lo.append(d["pct"] - d["ci"][0]); hi.append(d["ci"][1] - d["pct"])
        xx = x + (j - 2.5) * w
        ax.bar(xx, v, w, color=ST.C[m], ec="black", lw=0.5, label=lab)
        ax.errorbar(xx, v, yerr=[lo, hi], fmt="none", ecolor="black", capsize=2, lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([l for *_, l in conds]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Частка треків і ділянок, %", labelpad=8); ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_compare.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_compare()
    fig_rules()
    fig2()


def fig_meas():
    """Fig. 6 with the strict measure (reports/v5_strict.json, a4_strict.py)."""
    J = json.load(open(os.path.join(REPORTS, "v5_strict.json")))
    fig, (a, b) = plt.subplots(2, 1, figsize=(ST.W_IN, 7.8))
    bird = [("tbd", "СЗВ\n640"), ("tbd1920", "СЗВ\n1920"), ("att_none", "У0"), ("att_self", "УС"),
            ("att_ext1Hz", "УЗ-1")]
    cols = [ST.C["tbd"], ST.C["tbd"], ST.C["att_none"], ST.C["att_self"], ST.C["att_ext1Hz"]]
    v = [J[f"bird_{m}"]["pct"] for m, _ in bird] + [100.0]
    lo = [J[f"bird_{m}"]["pct"] - J[f"bird_{m}"]["ci"][0] for m, _ in bird] + [100 - 92.9]
    hi = [J[f"bird_{m}"]["ci"][1] - J[f"bird_{m}"]["pct"] for m, _ in bird] + [0]
    x = np.arange(len(v))
    a.bar(x, v, 0.6, color=cols + ["white"], ec="black", lw=0.6)
    a.errorbar(x, v, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=0.7)
    a.set_xticks(x); a.set_xticklabels([l for _, l in bird] + ["еталон"]); a.grid(axis="x", visible=False)
    a.set_ylim(0, 105); a.set_ylabel("Частка треків, %", labelpad=8); ST.panel(a, "а")
    w = 0.26
    for j, (m, lab) in enumerate((("tbd", "СЗВ"), ("att_none", "У0"), ("att_ext1Hz", "УЗ-1"))):
        d = [J[f"uav_{S}_{m}"] for S in (6, 9, 12)]
        xx = np.arange(3) + (j - 1) * w
        b.bar(xx, [q["pct"] for q in d], w, color=ST.C[m], ec="black", lw=0.6, label=lab)
        b.errorbar(xx, [q["pct"] for q in d], yerr=[[q["pct"] - q["ci"][0] for q in d], [q["ci"][1] - q["pct"] for q in d]],
                   fmt="none", ecolor="black", capsize=3, lw=0.7)
    b.set_xticks(np.arange(3)); b.set_xticklabels(["6 пкс", "9 пкс", "12 пкс"]); b.grid(axis="x", visible=False)
    b.set_ylim(0, 105); b.set_ylabel("Частка ділянок, %", labelpad=8); ST.panel(b, "б")
    b.legend(loc="upper left", ncol=1)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(FIG, "fig_meas.png"))
    plt.close(fig)

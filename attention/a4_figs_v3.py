#!/usr/bin/env python3
"""Figures of draft v3: measurability by strategy (birds, UAVs) and the decision-rule trade-off.
Numbers are recomputed from cache/e3_series.pkl, cache/e5_series.pkl, cache/truth_series_birds.pkl,
cache/det_midway_1920.pkl. Output: fig/fig_meas.png, fig/fig_rules.png, reports/v3_numbers.json"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import A4, CACHE, REPORTS, WIN, load   # noqa: E402
import a4_experiments as X                         # noqa: E402
import a4_rules as RR                              # noqa: E402

TNR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
font_manager.fontManager.addfont(TNR)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def longest(s):
    ok = ~np.isnan(np.array(s, float)[:, 0]); b = c = 0
    for v in ok:
        c = c + 1 if v else 0; b = max(b, c)
    return b


def tbd_native(thr):
    det = load(os.path.join(CACHE, "det_midway_1920.pkl")); tr = load(os.path.join(CACHE, "eval_tracks.pkl"))
    X.FLASH = thr; out = []
    for t in tr:
        pts = t["pts"]; o = X.TBD((pts[0][1], pts[0][2])); on = []
        for p in pts:
            o.step(det.get(int(p[0]), [])); ps = o.pos[-1]
            on.append(ps is not None and np.hypot(ps[0] - p[1], ps[1] - p[2]) < 25)
        out.append((o.series, on))
    X.FLASH = 0.25
    return out


def main():
    B = load(os.path.join(CACHE, "e3_series.pkl")); U = load(os.path.join(CACHE, "e5_series.pkl"))
    T = load(os.path.join(CACHE, "truth_series_birds.pkl"))
    N = {}
    bird_groups = [("СЗВ\n640", [B[k] for k in B if k[1] == "tbd"]),
                   ("СЗВ\n1920", tbd_native(0.1)),
                   ("У0", [B[k] for k in B if k[1] == "att_none"]),
                   ("УС", [B[k] for k in B if k[1] == "att_self"]),
                   ("УЗ-1", [B[k] for k in B if k[1] == "att_ext1Hz"]),
                   ("еталон", [(s, [True] * len(s)) for s in T.values()])]
    bm = []
    for lab, items in bird_groups:
        n = len(items); k = sum(longest(s) >= WIN for s, o in items)
        bm.append((lab, 100 * k / n, wilson(k, n))); N[f"bird_meas_{lab}"] = [k, n]
    um = {}
    for S in (6, 9, 12):
        for m, lab in (("tbd", "СЗВ"), ("att_none", "У0"), ("att_ext1Hz", "УЗ-1")):
            items = [v[k] for key, v in U.items() if key[2] == S for k in v if k[0] == m]
            n = len(items); k = sum(longest(s) >= WIN for s, o in items)
            um[(S, lab)] = (100 * k / n, wilson(k, n)); N[f"uav_meas_{S}_{lab}"] = [k, n]

    fig, (a, b) = plt.subplots(2, 1, figsize=(6.5, 7.6))
    x = np.arange(len(bm))
    a.bar(x, [v for _, v, _ in bm], color=["0.75", "0.75", "0.3", "0.3", "0.3", "white"], ec="black", lw=0.8)
    a.errorbar(x, [v for _, v, _ in bm], yerr=[[v - ci[0] for _, v, ci in bm], [ci[1] - v for _, v, ci in bm]],
               fmt="none", ecolor="black", capsize=3, lw=0.8)
    a.set_xticks(x); a.set_xticklabels([l for l, _, _ in bm])
    a.set_ylim(0, 105); a.set_ylabel("Частка треків, %", labelpad=8); a.set_title("а", loc="left")
    a.spines[["top", "right"]].set_visible(False)
    w = 0.26
    for j, (lab, col) in enumerate((("СЗВ", "0.75"), ("У0", "0.3"), ("УЗ-1", "0.55"))):
        vals = [um[(S, lab)][0] for S in (6, 9, 12)]
        lo = [um[(S, lab)][0] - um[(S, lab)][1][0] for S in (6, 9, 12)]
        hi = [um[(S, lab)][1][1] - um[(S, lab)][0] for S in (6, 9, 12)]
        xx = np.arange(3) + (j - 1) * w
        b.bar(xx, vals, w, color=col, ec="black", lw=0.8, label=lab)
        b.errorbar(xx, vals, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, lw=0.8)
    b.set_xticks(np.arange(3)); b.set_xticklabels(["6 пкс", "9 пкс", "12 пкс"])
    b.set_ylim(0, 105); b.set_ylabel("Частка ділянок, %", labelpad=8); b.set_title("б", loc="left")
    b.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(A4, "fig", "fig_meas.png"), dpi=300)
    plt.close(fig)

    rules = [("any", "будь-яке\nвікно"), ("pair", "два вікна\nпоспіль"), ("major", "більшість\nвікон"), ("full", "увесь\nряд")]
    bt, ba, uf = [], [], []
    for r, _ in rules:
        res = [RR.decide(s, [True] * len(s), 29.97, r) for s in T.values()]
        bt.append(100 * np.mean([q[0] for q in res]))
        it = [B[k] for k in B if k[1] == "att_none"]
        res = [RR.decide(s, o, 29.97, r) for s, o in it]
        ba.append(100 * np.mean([q[1] for q in res]))
        it = [v[k] for key, v in U.items() for k in v if k[0] in ("att_none", "att_ext1Hz")]
        res = [RR.decide(s, o, 25.0, r) for s, o in it]
        uf.append(100 * np.mean([q[0] for q in res]))
        N[f"rule_{r}"] = dict(bird_truth=bt[-1], bird_att_on_target=ba[-1], uav_false_bird=uf[-1], uav_n=len(it))
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    x = np.arange(len(rules)); w = 0.26
    ax.bar(x - w, bt, w, color="white", ec="black", lw=0.8, label="птахи, еталонне утримання")
    ax.bar(x, ba, w, color="0.3", ec="black", lw=0.8, label="птахи, утримання У0")
    ax.bar(x + w, uf, w, color="0.75", ec="black", lw=0.8, hatch="//", label="БпЛА: хибне рішення «птах»")
    for xi, v in zip(x + w, uf):
        ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in rules])
    ax.set_ylabel("Частка, %", labelpad=8); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right", fontsize=14)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(A4, "fig", "fig_rules.png"), dpi=300)
    plt.close(fig)
    json.dump(N, open(os.path.join(REPORTS, "v3_numbers.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(N, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

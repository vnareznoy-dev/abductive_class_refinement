#!/usr/bin/env python3
"""Failure example for 4b (reviewer 1, other remark 4), main variant 2 L_t.

Bird track 31 of the Midway video, strategy S0 (no hint), Python twin with A4_ALP_MULT=2. A large neighbouring bird
crosses the predicted position; its silhouette lies inside 2 L_t and nearer to the prediction than the target, so
the minimal explanation (3) accepts it and the track passes to the neighbour; without a hint it does not return.
Found by scanning the E26b alp_c2 S0 bird runs for a held on-target stretch of >= 20 frames followed by >= 15 held
frames farther than 3 s from the reference; track 118 was rejected (the reference itself jumps to a passing bird), track 21 is a similar crossing but with
L_t = 22 px, where 2 L_t exceeds the 30 px search gate; track 31 (L_t = 9 px, limit 18 px) is drawn.
Panels: a - frames before and after the transition (green cross - reference target, red x - track position,
dashed circle - limit 2 L_t around the prediction); b - distance from the track to the reference in target sizes s,
with the tolerance 2s of the metric.
Outputs: fig/fig_failure_c2.png (Ukrainian), fig/en/fig_failure_c2_en.png, reports/fig_failure_c2.txt"""
import os
import sys

os.environ["A4_ALP_MULT"] = "2"
import numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_style as ST                                         # noqa: E402
from a4_style import plt                                      # noqa: E402
from matplotlib.patches import Circle                         # noqa: E402
import a4_experiments as X                                    # noqa: E402
from common import A4, CACHE, MIDWAY, REPORTS, frames, load   # noqa: E402

TID, SHOW, HALF = int(os.environ.get("FAIL_TID", 31)), tuple(int(x) for x in os.environ.get("FAIL_SHOW", "1510,1511,1512,1513,1514,1515").split(",")), 45
TXT = {"ua": dict(fr="кадр", x="Номер кадру треку", y="Відстань до цілі, s", tol="допуск 2s", trans="перехід на сусіда",
                  ref="ціль (еталон)", pos="положення треку", lim="межа 2$L_t$", p=("а", "б"), out="fig_failure_c2.png"),
       "en": dict(fr="frame", x="Frame number of the track", y="Distance to the target, s", tol="tolerance 2s", trans="transition to the neighbour",
                  ref="target (reference)", pos="track position", lim="limit 2$L_t$", p=("a", "b"), out=os.path.join("en", "fig_failure_c2_en.png"))}


def run():
    T = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}[TID]
    s = float(np.median([max(p[3], p[4]) for p in T["pts"]]))
    truth = {int(p[0]): (p[1], p[2]) for p in T["pts"]}; ks = sorted(truth)
    h = X.Holder(truth[ks[0]], "none", alp=True); rows = []
    for k, fr in frames(MIDWAY, ks[0], ks[-1]):
        if k not in truth:
            continue
        L = float(np.median(h.sizes)) if len(h.sizes) >= X.ID_MIN else None
        h.step(fr, truth[k], None, [])
        tr = h.trace[-1]; p = h.pos[-1]
        keep = k in SHOW
        y0, x0 = int(truth[k][1]) - HALF, int(truth[k][0]) - HALF
        rows.append(dict(k=k, truth=truth[k], pos=p, cpos=h.last.copy() if h.last is not None else None, pred=tr["pred"], state=tr["state"], L=L,
                         crop=fr[max(y0, 0):y0 + 2 * HALF, max(x0, 0):x0 + 2 * HALF, ::-1].copy() if keep else None, x0=x0, y0=y0))
    return rows, s


def main():
    rows, s = run()
    d = np.array([np.nan if r["cpos"] is None else np.hypot(r["cpos"][0] - r["truth"][0], r["cpos"][1] - r["truth"][1]) for r in rows]) / s
    ki = [r["k"] for r in rows]; t0 = ki.index(SHOW[3])
    rep = [f"track {TID}: s = {s:.1f} px; frames {len(rows)}; transition at video frame {SHOW[3]} (track frame {t0}); "
           f"distance before {np.nanmean(d[t0 - 10:t0]):.2f} s, after {np.nanmean(d[t0:t0 + 15]):.2f} s; "
           f"L_t at transition {rows[t0]['L']}; states after: {[r['state'] for r in rows[t0:t0 + 6]]}"]
    open(os.path.join(REPORTS, "fig_failure_c2.txt"), "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
    for lang, tx in TXT.items():
        fig = plt.figure(figsize=(ST.W_IN, 4.2))
        gs = fig.add_gridspec(2, len(SHOW), height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.08)
        for j, k in enumerate(SHOW):
            r = rows[ki.index(k)]; ax = fig.add_subplot(gs[0, j]); ax.grid(False)
            ax.imshow(r["crop"], extent=(r["x0"], r["x0"] + r["crop"].shape[1], r["y0"] + r["crop"].shape[0], r["y0"]), interpolation="nearest")
            ax.set_xlim(r["x0"], r["x0"] + 2 * HALF); ax.set_ylim(r["y0"] + 2 * HALF, r["y0"]); ax.set_xticks([]); ax.set_yticks([])
            if r["pred"] is not None and r["L"]:
                ax.add_patch(Circle(r["pred"], 2 * r["L"], fill=False, ec=ST.YELLOW, lw=1.2, ls="--"))
            ax.plot(*r["truth"], "+", color=ST.GREEN, ms=9, mew=1.8)
            if r["pos"] is not None:
                ax.plot(*r["pos"], "x", color=ST.ORANGE, ms=8, mew=1.8)
            ax.set_xlabel(f"{tx['fr']} {k - rows[0]['k']}", fontsize=12, labelpad=2)
            if j == 0:
                ST.panel(ax, tx["p"][0])
        ax = fig.add_subplot(gs[1, :])
        x = np.arange(len(rows))
        ax.plot(x, d, color=ST.ORANGE, lw=1.3)
        ax.axhline(2, color="black", ls="--", lw=1.0)
        ax.axvline(t0, color=ST.GREY, ls=":", lw=1.2)
        ax.text(t0 - 1.5, np.nanmax(d) * 0.75, tx["trans"], fontsize=12, va="top", ha="right")
        ax.text(1, 2.3, tx["tol"], fontsize=12, va="bottom")
        ax.set_xlim(0, len(rows) - 1); ax.set_ylim(0, np.nanmax(d) * 1.05)
        ax.set_xlabel(tx["x"], labelpad=3); ax.set_ylabel(tx["y"], labelpad=6); ST.panel(ax, tx["p"][1]); ST.comma(ax)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=ST.GREEN, marker="+", ls="", ms=9, mew=1.8), Line2D([], [], color=ST.ORANGE, marker="x", ls="", ms=8, mew=1.8),
             Line2D([], [], color=ST.YELLOW, ls="--", lw=1.2)]
        ST.top_legend(fig, h, [tx["ref"], tx["pos"], tx["lim"]], 3, y=0.92)
        fig.subplots_adjust(top=0.9, bottom=0.11, left=0.1, right=0.98)
        fig.savefig(os.path.join(A4, "fig", tx["out"]), dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()

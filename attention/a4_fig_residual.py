#!/usr/bin/env python3
"""Figure for the revision of 4b (reviewer 1, remark 1): distribution of the residual
"silhouette centre - prediction (1)" in body lengths, r / L_t, with the limits 1 L_t and 2 L_t.

  a) birds: reference tracks (E25, cache/e25_residual.pkl) and silhouettes accepted on target by the
     holder without constraint (2) (E26 base variant, a4_rev_tables.residuals_base);
  b) UAV 6 / 9 / 12 px: silhouettes accepted on target (E26 base). The UAV reference boxes are
     not shown: annotation boxes are smooth (residual median 0.03 L_t) and say nothing about the
     silhouette.
Histogram of r / L_t, bin 0.1, share of frames in %, logarithmic axis; values above 4 L_t are
counted in the last bin. The share of frames beyond 1 L_t and 2 L_t is printed in each panel.
Outputs: fig/fig_residual.png (Ukrainian), fig/en/fig_residual_en.png, reports/fig_residual.txt"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_style as ST                      # noqa: E402
from a4_style import plt                   # noqa: E402
from common import A4, CACHE, REPORTS, load   # noqa: E402
import a4_rev_tables as T                  # noqa: E402

XMAX, BIN = 4.0, 0.1
TXT = {
    "ua": dict(ref="птахи, еталонні треки", acc="птахи, прийняті силуети",
               u6="БпЛА 6 пкс", u9="БпЛА 9 пкс", u12="БпЛА 12 пкс",
               x="Нев'язка $|\\mathbf{c} - \\hat{\\mathbf{x}}|\\,/\\,L_t$",
               y="Частка кадрів, %", beyond="за межею", out="fig_residual.png"),
    "en": dict(ref="birds, reference tracks", acc="birds, accepted silhouettes",
               u6="UAV 6 px", u9="UAV 9 px", u12="UAV 12 px",
               x="Residual $|\\mathbf{c} - \\hat{\\mathbf{x}}|\\,/\\,L_t$",
               y="Share of frames, %", beyond="beyond", out=os.path.join("en", "fig_residual_en.png")),
}


def ratios():
    E = load(os.path.join(CACHE, "e25_residual.pkl"))
    ref = np.concatenate([np.array(v)[:, 0] / np.array(v)[:, 1] for v in E["bird"].values()])
    R = load(os.path.join(CACHE, "e26_runs.pkl"))
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    acc = {g: np.array([r / lt for r, lt in rows]) for g, rows in T.residuals_base(R, size).items()}
    return {"ref": ref, "acc": acc["bird"], "u6": acc["uav 6"], "u9": acc["uav 9"], "u12": acc["uav 12"]}


def hist(ax, x, color, label, ls="-"):
    bins = np.arange(0, XMAX + BIN / 2, BIN)
    h, _ = np.histogram(np.clip(x, 0, XMAX - 1e-9), bins=bins)
    y = 100 * h / len(x)
    ax.stairs(np.where(y > 0, y, np.nan), bins, color=color, lw=1.5, ls=ls, label=label)


def main():
    D = ratios()
    rep = ["Residual r / L_t: share of frames beyond c L_t, %"]
    for k, v in D.items():
        rep.append(f"  {k:4s} n={len(v):6d} median {np.median(v):.3f}  > 1 L_t {100*np.mean(v > 1):.1f}  "
                   f"> 1.5 L_t {100*np.mean(v > 1.5):.1f}  > 2 L_t {100*np.mean(v > 2):.1f}")
    open(os.path.join(REPORTS, "fig_residual.txt"), "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
    for lang, t in TXT.items():
        fig, axes = plt.subplots(1, 2, figsize=(ST.W_IN, 3.9), sharey=True)
        a, b = axes
        hist(a, D["ref"], ST.BLUE, t["ref"]); hist(a, D["acc"], ST.ORANGE, t["acc"])
        hist(b, D["u6"], ST.PURPLE, t["u6"]); hist(b, D["u9"], ST.GREEN, t["u9"]); hist(b, D["u12"], ST.YELLOW, t["u12"])
        for ax, keys, letter in ((a, ("ref", "acc"), "а" if lang == "ua" else "a"), (b, ("u6", "u9", "u12"), "б" if lang == "ua" else "b")):
            ax.set_yscale("log"); ax.set_ylim(0.005, 100); ax.set_xlim(0, XMAX)
            for xv in (1, 2):
                ax.axvline(xv, color="black", lw=1.0, ls="--" if xv == 1 else "-")
                ax.text(xv + 0.05, 75, f"{xv}$L_t$" if xv == 2 else "$L_t$", fontsize=13, va="top")
            vals = [D[k] for k in keys]
            s1 = " / ".join(f"{100*np.mean(v > 1):.1f}".replace(".", ",") for v in vals)
            s2 = " / ".join(f"{100*np.mean(v > 2):.1f}".replace(".", ",") for v in vals)
            ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel(t["x"], labelpad=4); ST.panel(ax, letter); ST.comma(ax, y=False)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
        a.set_ylabel(t["y"], labelpad=6)
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
        ST.top_legend(fig, h1 + h2, l1 + l2, 3, y=0.86)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(os.path.join(A4, "fig", t["out"]), bbox_inches="tight", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()

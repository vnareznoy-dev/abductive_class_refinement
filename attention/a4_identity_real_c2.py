#!/usr/bin/env python3
"""Fig. 3 of 4b for the main variant 2 L_t (revision of a4_identity_real.py / a4_identity_real_en.py).

With the limit 2 L_t the old frame (track 17, L_t = 23 px) no longer shows a refutation: 2 L_t = 46 px exceeds
the 30 px search gate. This script runs the Python twin with A4_ALP_MULT=2 (strategy SE-1, seed 7, as Fig. 4) on
all 50 bird tracks in one pass over the Midway video and collects the frames in which
  * a candidate silhouette lies inside the search gate but farther than 2 L_t from the prediction (refuted),
  * the accepted silhouette is the target itself (within 6 px of the reference),
ranked by: a classifier box on the refuted silhouette, then the refuted candidate being a separate object
(farther than 1.5 L_t from the target's reference centre). The chosen frame is drawn in Ukrainian and English.
Outputs: fig/fig1_identity_real_c2.png, fig/en/fig1_identity_real_c2_en.png, reports/identity_real_c2.txt"""
import os
import sys

os.environ["A4_ALP_MULT"] = "2"
import numpy as np                                            # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_experiments as X                                    # noqa: E402
from common import A4, CACHE, MIDWAY, REPORTS, frames, load   # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 14})
SEED, HALF, M = 7, 40, X.ALP_MULT
TXT = {
    "ua": dict(a="а) кадр треку птаха (відеозапис атолу Мідвей)", b="б) гіпотези тотожності", unseen="unseen (ціль невидима)",
               ic="обмеження цілісності:\n", px=" пкс", adm1="допустима", admn="допустимі", ref="спростована", andw=" і ",
               minx="мінімальне пояснення:\nнайменше відхилення " + r"$\rightarrow$ ціль $= b_1$",
               lab=r"мітки лише від $b_1$ $\rightarrow$" + "\nабдуктивне уточнення класу",
               note="якщо всі same(b)\nспростовано –\nабдукується unseen:\nкадр без мітки", out="fig1_identity_real_c2.png"),
    "en": dict(a="a) frame of a bird track (Midway Atoll video)", b="b) identity hypotheses", unseen="unseen (target not visible)",
               ic="integrity constraint:\n", px=" px", adm1="admissible", admn="admissible", ref="refuted", andw=" and ",
               minx="minimal explanation:\nsmallest deviation " + r"$\rightarrow$ target $= b_1$",
               lab=r"labels only from $b_1$ $\rightarrow$" + "\nabductive class refinement",
               note="if all same(b)\nare refuted,\nunseen is abduced:\nframe without a label", out=os.path.join("en", "fig1_identity_real_c2_en.png")),
}


def run():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl")); det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    hs, rows, nlog = {}, [], {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            tx, ty = truth[tid][k]
            if tid not in hs:
                hs[tid] = X.Holder((tx, ty), "ext", period=int(X.FPS_MIDWAY), sigma=15.0, seed=SEED, alp=True); nlog[tid] = 0
                hs[tid].n_frame = 0
            h = hs[tid]
            near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < X.GATE_DET]
            fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
            h.step(fr, (tx, ty), None if fl is None else (fl[0], fl[1]), det.get(k, []))
            if len(h.alp_log) > nlog[tid]:
                nlog[tid] = len(h.alp_log)
                pred, cands, L, chosen = h.alp_log[-1]
                if L is None or chosen is None:
                    continue
                px, py = pred; lim = M * max(L, X.ALP_FLOOR)
                far = [c for c in cands if np.hypot(c[0] - px, c[1] - py) > lim]
                ch = cands[chosen]
                if not far or np.hypot(ch[0] - tx, ch[1] - ty) > 6:
                    continue
                y0, x0 = int(py) - HALF, int(px) - HALF
                crop = fr[max(y0, 0):y0 + 2 * HALF, max(x0, 0):x0 + 2 * HALF, ::-1].copy()
                rows.append(dict(tid=tid, k=k, pred=pred, cands=cands, L=L, chosen=chosen, far=far, crop=crop, x0=x0, y0=y0,
                                 dets=[d for d in det.get(k, []) if d[4] >= X.FLASH], truth=(tx, ty)))
    return rows


def score(r):
    px, py = r["pred"]; tx, ty = r["truth"]
    sep = [c for c in r["far"] if np.hypot(c[0] - tx, c[1] - ty) > 1.5 * r["L"]]
    det_on = any(np.hypot(d[0] - c[0], d[1] - c[1]) < 8 for c in sep for d in r["dets"])
    margin = min(np.hypot(c[0] - px, c[1] - py) for c in r["far"]) - M * r["L"]
    return (det_on, bool(sep), min(margin, 6.0), r["L"])


def draw(r, t):
    px, py = r["pred"]; L = r["L"]; lim = M * max(L, X.ALP_FLOOR); x0, y0 = r["x0"], r["y0"]; crop = r["crop"]
    far = sorted(r["far"], key=lambda c: -np.hypot(c[0] - r["truth"][0], c[1] - r["truth"][1]))   # b2 = the separate object
    fig = plt.figure(figsize=(11, 5.4))
    ax = fig.add_axes([0.02, 0.06, 0.40, 0.88])
    ax.imshow(crop, extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0), interpolation="nearest")
    ax.set_xlim(x0, x0 + 2 * HALF); ax.set_ylim(y0 + 2 * HALF, y0); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Circle((px, py), lim, fill=False, ec="#2e8b57", lw=2.2, ls="--"))
    ax.plot([px - 2.5, px + 2.5], [py, py], color="#d62728", lw=2.4); ax.plot([px, px], [py - 2.5, py + 2.5], color="#d62728", lw=2.4)
    ax.text(px - 3.5, py - 3.5, r"$\hat{x}_t$", color="#d62728", fontsize=18, ha="right")
    ch = r["cands"][r["chosen"]]; s = max(ch[2], 3) / 2 + 1.5
    ax.add_patch(Rectangle((ch[0] - s, ch[1] - s), 2 * s, 2 * s, fill=False, ec="#2ca02c", lw=2.4))
    ax.text(ch[0] + s + 0.8, ch[1] - s, r"$b_1$", fontsize=18, color="#1b6e1b", va="bottom")
    c = far[0]; s2 = max(c[2], 3) / 2 + 1.5
    ax.add_patch(Rectangle((c[0] - s2, c[1] - s2), 2 * s2, 2 * s2, fill=False, ec="#c0392b", lw=2.4, ls="--"))
    ax.text(c[0] + s2 + 0.8, c[1] + s2, r"$b_2$", fontsize=18, color="#c0392b", va="top")
    adm = sorted([c for j, c in enumerate(r["cands"]) if j != r["chosen"] and np.hypot(c[0] - px, c[1] - py) <= lim],
                 key=lambda c: np.hypot(c[0] - px, c[1] - py))
    for m, c3 in enumerate(adm):
        s3 = max(c3[2], 3) / 2 + 1.5
        ax.add_patch(Rectangle((c3[0] - s3, c3[1] - s3), 2 * s3, 2 * s3, fill=False, ec="0.35", lw=2.0, ls=":"))
        ax.text(c3[0] - s3 - 0.8, c3[1] - s3, r"$b_%d$" % (m + 3), fontsize=18, color="0.25", va="bottom", ha="right")
    for d in r["dets"]:
        if x0 <= d[0] <= x0 + 2 * HALF and y0 <= d[1] <= y0 + 2 * HALF:
            ax.add_patch(Rectangle((d[0] - d[2] / 2, d[1] - d[3] / 2), d[2], d[3], fill=False, ec="#e6b800", lw=1.6))
    ax.annotate("", xy=(px - lim, py), xytext=(px, py), arrowprops=dict(arrowstyle="<->", color="#2e8b57", lw=1.5))
    ax.text(px - 0.78 * lim, py - 1.2, r"$2L_t$", color="#2e8b57", fontsize=17, ha="center", va="bottom")
    ax.set_title(t["a"], fontsize=16)
    bx = fig.add_axes([0.45, 0.02, 0.55, 0.96]); bx.axis("off"); bx.set_xlim(5.6, 13); bx.set_ylim(0, 6.2)

    def box(x, y, w, h, s_, fc, ec, fs=15):
        bx.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.6))
        bx.text(x, y, s_, ha="center", va="center", fontsize=fs)

    def arr(a, b_, c_, d_, col="0.15"):
        bx.annotate("", xy=(c_, d_), xytext=(a, b_), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=col))
    f1 = lambda v: ("%.1f" % v).replace(".", ",") if t is TXT["ua"] else "%.1f" % v
    d1 = np.hypot(ch[0] - px, ch[1] - py); d2 = np.hypot(far[0][0] - px, far[0][1] - py); dd = [np.hypot(c3[0] - px, c3[1] - py) for c3 in adm]
    bx.text(7.3, 5.75, t["b"], fontsize=16, ha="center")
    box(7.3, 5.0, 2.4, 0.62, r"same($b_1$)", "#e8f5e9", "#2e8b57")
    box(7.3, 4.15, 2.4, 0.62, r"same($b_2$)", "#fdecea", "#c0392b")
    names = ", ".join(r"same($b_%d$)" % (m + 3) for m in range(len(adm)))
    if adm:
        box(7.3, 3.3, 2.4, 0.62, names, "#f3f3f3", "0.4")
    box(7.3, 2.45, 2.4, 0.62, t["unseen"].replace(" (", "\n("), "#f3f3f3", "0.5", fs=13)
    box(10.9, 4.55, 3.6, 1.0, t["ic"] + r"$\|c_b-\hat{x}_t\| \leq 2L_t$", "white", "0.3")
    arr(8.5, 5.0, 9.1, 4.75); arr(8.5, 4.15, 9.1, 4.45)
    if adm:
        arr(8.5, 3.3, 9.1, 4.2)
    bx.text(10.9, 3.75, r"$L_t$ = " + f1(L) + t["px"] + r";  $2L_t$ = " + f1(lim) + t["px"], fontsize=14, ha="center")
    bx.text(10.9, 3.35, r"same($b_1$): " + f1(d1) + " ≤ " + f1(lim) + ", " + t["adm1"], fontsize=14, color="#2e8b57", ha="center")
    if adm:
        bx.text(10.9, 2.95, names + ": " + t["andw"].join(f1(x) for x in dd) + " ≤ " + f1(lim) + ", " + (t["admn"] if len(adm) > 1 else t["adm1"]),
                fontsize=14, color="0.3", ha="center")
    bx.text(10.9, 2.55, r"same($b_2$): " + f1(d2) + " > " + f1(lim) + ", " + t["ref"], fontsize=14, color="#c0392b", ha="center")
    box(10.9, 1.65, 3.6, 0.75, t["minx"], "#e8f5e9", "#2e8b57", fs=14); arr(10.9, 2.35, 10.9, 2.05, "#2e8b57")
    box(10.9, 0.55, 3.6, 0.8, t["lab"], "#eaf1fb", "#3a64a0", fs=14); arr(10.9, 1.27, 10.9, 0.97, "#3a64a0")
    bx.text(7.3, 1.35, t["note"], fontsize=13.5, ha="center", va="center", color="0.3")
    fig.savefig(os.path.join(A4, "fig", t["out"]), dpi=300, bbox_inches="tight"); plt.close(fig)
    return d1, d2, dd, lim


def main():
    cache = os.path.join(CACHE, "fig3_c2_candidates.pkl")
    import pickle
    if os.path.exists(cache) and os.environ.get("FIG3_PICK"):
        rows = pickle.load(open(cache, "rb"))
    else:
        rows = run(); rows.sort(key=score, reverse=True)
    L = [f"Fig. 3 (2 L_t): {len(rows)} frames with a refuted candidate and the target accepted; top 8 by score:"]
    for r in rows[:8]:
        L.append(f"  track {r['tid']} video frame {r['k']} L_t {r['L']:.1f} score {score(r)}")
    pick = int(os.environ.get("FIG3_PICK", "0")); r = rows[pick]
    for t in TXT.values():
        d1, d2, dd, lim = draw(r, t)
    L.append(f"drawn: track {r['tid']} video frame {r['k']}; L_t {r['L']:.2f}, 2L_t {lim:.2f}; b1 {d1:.2f}; admissible {[round(float(x), 2) for x in dd]}; refuted b2 {d2:.2f}")
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "identity_real_c2.txt"), "w").write(t + "\n")
    if not os.environ.get("FIG3_PICK"):
        pickle.dump([{k: v for k, v in r.items()} for r in rows[:30]], open(cache, "wb"))


if __name__ == "__main__":
    main()

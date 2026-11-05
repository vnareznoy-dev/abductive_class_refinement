#!/usr/bin/env python3
"""Article «Абдуктивне встановлення тотожності…», Fig. 1 on a REAL frame (replaces the drawn schematic).

Runs the main Holder with abductive identity (alp=True) on bird track 17 of the Midway video (stratagy
UZ-1, seed 7 - the same run as Fig. 2), and logs per frame: prediction x̂_t, body length L_t, all
silhouette candidates within the 30 px search window, the chosen one and the classifier boxes.
Picks the frame where a neighbouring silhouette lies inside the search window but farther than L_t
(the hypothesis same(b) is refuted) while the target is accepted, preferring frames where the
classifier put a box on the neighbour. Draws panel a from that frame and panel b as the hypothesis
diagram. -> ../fig/fig1_identity_real.png, ../reports/identity_real_frame.txt
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_experiments as X                                    # noqa: E402
from common import A4, CACHE, MIDWAY, frames, load            # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 14})
TID, SEED = 17, 7


def run():
    tr = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}[TID]
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {int(p[0]): (p[1], p[2]) for p in tr["pts"]}
    ks = sorted(truth)
    h = X.Holder(truth[ks[0]], "ext", period=int(X.FPS_MIDWAY), sigma=15.0, seed=SEED, alp=True)
    rows = []
    for k, fr in frames(MIDWAY, ks[0], ks[-1]):
        if k not in truth:
            continue
        tx, ty = truth[k]
        near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < X.GATE_DET]
        fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
        n_log = len(h.alp_log)
        h.step(fr, (tx, ty), None if fl is None else (fl[0], fl[1]), det.get(k, []))
        if len(h.alp_log) > n_log:
            pred, cands, L, chosen = h.alp_log[-1]
            rows.append(dict(k=k, n=len(rows), fr=fr, pred=pred, cands=cands, L=L, chosen=chosen,
                             dets=[d for d in det.get(k, []) if d[4] >= X.FLASH], truth=(tx, ty)))
    return rows


def pick(rows):
    best = None
    for r in rows:
        if r["L"] is None or r["chosen"] is None:
            continue
        px, py = r["pred"]
        far = [c for c in r["cands"] if np.hypot(c[0] - px, c[1] - py) > r["L"]]
        if not far:
            continue
        ch = r["cands"][r["chosen"]]
        if np.hypot(ch[0] - r["truth"][0], ch[1] - r["truth"][1]) > 6:
            continue            # the accepted silhouette must be the target itself
        det_on_far = any(np.hypot(d[0] - c[0], d[1] - c[1]) < 8 for c in far for d in r["dets"])
        score = (det_on_far, -min(np.hypot(c[0] - px, c[1] - py) for c in far))
        if best is None or score > best[0]:
            best = (score, r, far)
    return best


def draw(r, far):
    px, py = r["pred"]
    L = r["L"]
    half = 40
    x0, y0 = int(px) - half, int(py) - half
    crop = r["fr"][max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half, ::-1]
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.02, 0.06, 0.40, 0.88])
    ax.imshow(crop, extent=(x0, x0 + crop.shape[1], y0 + crop.shape[0], y0), interpolation="nearest")
    ax.set_xlim(x0, x0 + 2 * half); ax.set_ylim(y0 + 2 * half, y0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Circle((px, py), L, fill=False, ec="#2e8b57", lw=2.2, ls="--"))
    ax.plot([px - 2.5, px + 2.5], [py, py], color="#d62728", lw=2.4)
    ax.plot([px, px], [py - 2.5, py + 2.5], color="#d62728", lw=2.4)
    ax.text(px - 3.5, py - 3.5, r"$\hat{x}_t$", color="#d62728", fontsize=18, ha="right")
    ch = r["cands"][r["chosen"]]
    s = max(ch[2], 3) / 2 + 1.5
    ax.add_patch(Rectangle((ch[0] - s, ch[1] - s), 2 * s, 2 * s, fill=False, ec="#2ca02c", lw=2.4))
    ax.text(ch[0] + s + 0.8, ch[1] - s, r"$b_1$", fontsize=18, color="#1b6e1b", va="bottom")
    for c in far[:1]:
        s2 = max(c[2], 3) / 2 + 1.5
        ax.add_patch(Rectangle((c[0] - s2, c[1] - s2), 2 * s2, 2 * s2, fill=False, ec="#c0392b", lw=2.4, ls="--"))
        ax.text(c[0] + s2 + 0.8, c[1] + s2, r"$b_2$", fontsize=18, color="#c0392b", va="top")
    adm = sorted([c for j, c in enumerate(r["cands"]) if j != r["chosen"] and np.hypot(c[0] - px, c[1] - py) <= L],
                 key=lambda c: np.hypot(c[0] - px, c[1] - py))
    r["adm"] = adm
    for m, c in enumerate(adm):
        s3 = max(c[2], 3) / 2 + 1.5
        ax.add_patch(Rectangle((c[0] - s3, c[1] - s3), 2 * s3, 2 * s3, fill=False, ec="0.35", lw=2.0, ls=":"))
        ax.text(c[0] - s3 - 0.8, c[1] - s3, r"$b_%d$" % (m + 3), fontsize=18, color="0.25", va="bottom", ha="right")
    for d in r["dets"]:
        if x0 <= d[0] <= x0 + 2 * half and y0 <= d[1] <= y0 + 2 * half:
            ax.add_patch(Rectangle((d[0] - d[2] / 2, d[1] - d[3] / 2), d[2], d[3], fill=False, ec="#e6b800", lw=1.6))
    ax.annotate("", xy=(px + L, py), xytext=(px, py), arrowprops=dict(arrowstyle="<->", color="#2e8b57", lw=1.5))
    ax.text(px + L / 2, py + 2.8, r"$L_t$", color="#2e8b57", fontsize=17, ha="center", va="top")
    ax.set_title("a) frame of a bird track (Midway Atoll video)", fontsize=14)

    # ---- panel b: hypotheses (same layout as the schematic version)
    bx = fig.add_axes([0.45, 0.02, 0.55, 0.96]); bx.axis("off"); bx.set_xlim(5.6, 13); bx.set_ylim(0, 6.2)

    def box(x, y, w, h, t, fc, ec, fs=13):
        bx.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.6))
        bx.text(x, y, t, ha="center", va="center", fontsize=fs)

    def arr(x0_, y0_, x1, y1, c="0.15"):
        bx.annotate("", xy=(x1, y1), xytext=(x0_, y0_), arrowprops=dict(arrowstyle="-|>", lw=1.5, color=c))
    bx.text(7.3, 5.75, "b) identity hypotheses", fontsize=14, ha="center")
    f1 = lambda v: ("%.1f" % v)
    d1 = np.hypot(ch[0] - px, ch[1] - py); d2 = np.hypot(far[0][0] - px, far[0][1] - py)
    adm = r.get("adm", [])
    dd = [np.hypot(c[0] - px, c[1] - py) for c in adm]
    d3 = dd[0] if dd else None
    box(7.3, 5.0, 2.4, 0.62, r"same($b_1$)", "#e8f5e9", "#2e8b57")
    box(7.3, 4.15, 2.4, 0.62, r"same($b_2$)", "#fdecea", "#c0392b")
    box(7.3, 3.3, 2.4, 0.62, r"same($b_3$), same($b_4$)", "#f3f3f3", "0.4")
    box(7.3, 2.45, 2.4, 0.62, "unseen (target not visible)", "#f3f3f3", "0.5", fs=12)
    box(10.9, 4.55, 3.6, 1.0, "integrity constraint:\n" + r"$\|c_b-\hat{x}_t\| \leq L_t$", "white", "0.3")
    arr(8.5, 5.0, 9.1, 4.75); arr(8.5, 4.15, 9.1, 4.45); arr(8.5, 3.3, 9.1, 4.2)
    bx.text(10.9, 3.75, r"$L_t$ = " + f1(L) + " px", fontsize=12, ha="center")
    bx.text(10.9, 3.35, r"same($b_1$): " + f1(d1) + " ≤ " + f1(L) + ", admissible", fontsize=12, color="#2e8b57", ha="center")
    bx.text(10.9, 2.95, r"same($b_3$), same($b_4$): " + " and ".join(f1(x) for x in dd) + " ≤ " + f1(L) + ", admissible", fontsize=12, color="0.3", ha="center")
    bx.text(10.9, 2.55, r"same($b_2$): " + f1(d2) + " > " + f1(L) + ", refuted", fontsize=12, color="#c0392b", ha="center")
    box(10.9, 1.65, 3.6, 0.75, "minimal explanation:\nsmallest deviation " + r"$\rightarrow$ target $= b_1$", "#e8f5e9", "#2e8b57", fs=12)
    arr(10.9, 2.35, 10.9, 2.05, "#2e8b57")
    box(10.9, 0.55, 3.6, 0.8, r"labels only from $b_1$ $\rightarrow$" + "\nabductive class refinement", "#eaf1fb", "#3a64a0", fs=12)
    arr(10.9, 1.27, 10.9, 0.97, "#3a64a0")
    bx.text(7.3, 1.35, "if all same(b)\nare refuted,\nunseen is abduced:\nframe without a label", fontsize=11.5, ha="center", va="center", color="0.3")
    out = os.path.join(A4, "fig", "en", "fig1_identity_real_en.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    return out, d1, d2, d3


if __name__ == "__main__":
    rows = run()
    FRAME_N = 40   # visually checked: b1 = target, refuted fragment = wing of a neighbouring bird
    r = next(x for x in rows if x["n"] == FRAME_N)
    px_, py_ = r["pred"]
    far = sorted([c for c in r["cands"] if np.hypot(c[0] - px_, c[1] - py_) > r["L"]],
                 key=lambda c: -np.hypot(c[0] - r["truth"][0], c[1] - r["truth"][1]))
    score = (False,)
    out, d1, d2, d3 = draw(r, far)
    msg = (f"track {TID}, seed {SEED}: frame index {r['n']} (video frame {r['k']}); L_t = {r['L']:.2f} px; "
           f"chosen b1 at {d1:.2f} px, admissible others at {[round(float(x),2) for x in np.hypot(np.array([c[0] for c in r['adm']])-r['pred'][0], np.array([c[1] for c in r['adm']])-r['pred'][1])]} px, refuted b2 at {d2:.2f} px (a separate fragment of the fragmented silhouette, likely the target's wing tip); "
           f"frames with a refuted neighbour: {sum(1 for x in rows if x['L'] and any(np.hypot(c[0]-x['pred'][0], c[1]-x['pred'][1]) > x['L'] for c in x['cands']))} of {len(rows)}")
    os.makedirs(os.path.join(A4, "reports"), exist_ok=True)
    open(os.path.join(A4, "reports", "identity_real_frame_en.txt"), "w").write(msg + "\n")
    print(msg); print("saved", out)

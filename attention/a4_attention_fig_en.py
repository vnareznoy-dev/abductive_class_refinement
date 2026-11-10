#!/usr/bin/env python3
"""How the attention works on one target, frame by frame.

Runs the main Holder (a4_experiments.py) with its per-frame trace (prediction,
window, chosen silhouette, ladder state), checks that the held series equals the
cached series of the main run, takes the verdict from the primary decision rule
(a4_rules.decide, "full") and draws:
  а -- filmstrip of frames with overlays;
  б -- time line: classifier confidence, ladder state, held silhouette width w_t;
  в -- spectrum of the longest continuous held run with the decision criterion (6).
Outputs: fig/fig_attn_bird.png (Midway track 17, external cue 1 Hz, seed 7),
         fig/fig_attn_uav.png  (DUT video03, 9 px, no cue).
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import A4, CACHE, MIDWAY, WIN, band_test, frames, load, local_blobs   # noqa: E402
import a4_drones as D                                                          # noqa: E402
import a4_experiments as X                                                     # noqa: E402
import a4_rules as RR                                                          # noqa: E402

import a4_style as ST                                                          # noqa: E402
TNR = ST.TNR
ST.apply()
ST_NAME = {"hold": "track maint.", "coast": "extrapolation", "reacq": "re-acquisition", "lost": "lost"}


def run_holder(holder, frame_iter):
    """Steps the MAIN Holder of a4_experiments over (k, frame, truth_xy, flash_xy, dets)
    and joins its per-frame trace with the classifier response for drawing."""
    log = []
    for k, fr, txy, fxy, dets in frame_iter:
        holder.step(fr, txy, fxy, dets)
        t = holder.trace[-1]
        conf = max((d[4] for d in dets if np.hypot(d[0] - txy[0], d[1] - txy[1]) < X.GATE_DET), default=0.0)
        log.append(dict(k=k, fr=fr, pred=t["pred"], win=t["win"], pick=t["pick"], state=t["state"], conf=conf,
                        dets=[d for d in dets if d[4] >= X.FLASH], truth=txy,
                        on=holder.pos[-1] is not None and np.hypot(holder.pos[-1][0] - txy[0], holder.pos[-1][1] - txy[1]) < X.ON_TARGET))
    return log


def same(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def longest_run(log):
    best, cur, end = 0, 0, -1
    for i, e in enumerate(log):
        cur = cur + 1 if e["pick"] is not None else 0
        if cur > best:
            best, end = cur, i
    return end - best + 1, end + 1


def comma(ax):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: (f"{v:g}"))
    ax.xaxis.set_major_formatter(f); ax.yaxis.set_major_formatter(f)


def render_tile(e, crop, up=4):
    """BGR crop centred on truth with overlays, upscaled (nearest) by `up` -- for the GIF."""
    cx, cy = e["truth"]; x0, y0 = int(cx - crop / 2), int(cy - crop / 2)
    H, W = e["fr"].shape[:2]
    tile = np.full((crop, crop, 3), 255, np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + crop), min(H, y0 + crop)
    tile[ya - y0:yb - y0, xa - x0:xb - x0] = e["fr"][ya:yb, xa:xb]
    t = cv2.resize(tile, (crop * up, crop * up), interpolation=cv2.INTER_NEAREST)
    P = lambda x, y: (int((x - x0) * up), int((y - y0) * up))
    if e["win"] is not None:
        p, q = P(e["win"][0] - 60, e["win"][1] - 60), P(e["win"][0] + 60, e["win"][1] + 60)
        for i in range(p[0], q[0], 16):
            cv2.line(t, (i, p[1]), (min(i + 8, q[0]), p[1]), (255, 255, 255), 2); cv2.line(t, (i, q[1]), (min(i + 8, q[0]), q[1]), (255, 255, 255), 2)
        for i in range(p[1], q[1], 16):
            cv2.line(t, (p[0], i), (p[0], min(i + 8, q[1])), (255, 255, 255), 2); cv2.line(t, (q[0], i), (q[0], min(i + 8, q[1])), (255, 255, 255), 2)
    for d in e["dets"]:
        cv2.rectangle(t, P(d[0] - d[2] / 2, d[1] - d[3] / 2), P(d[0] + d[2] / 2, d[1] + d[3] / 2), (0, 220, 255), 2)
    if e["pick"] is not None:
        p = e["pick"]
        cv2.rectangle(t, P(p[0] - p[2] / 2 - 2, p[1] - p[3] / 2 - 2), P(p[0] + p[2] / 2 + 2, p[1] + p[3] / 2 + 2), (0, 255, 0), 2)
    if e["pred"] is not None:
        cv2.drawMarker(t, P(*e["pred"]), (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
    return t


def write_gif(log, crop, path, fps):
    from PIL import Image, ImageDraw, ImageFont
    fnt = ImageFont.truetype(TNR, 30)
    ims = []
    for i, e in enumerate(log):
        t = cv2.cvtColor(render_tile(e, crop, 3), cv2.COLOR_BGR2RGB)
        im = Image.new("RGB", (t.shape[1], t.shape[0] + 80), "white"); im.paste(Image.fromarray(t), (0, 0))
        dr = ImageDraw.Draw(im)
        dr.text((10, t.shape[0] + 6), f"frame {i}: {ST_NAME[e['state']]}", fill="black", font=fnt)
        dr.text((10, t.shape[0] + 42), f"classifier c = {e['conf']:.2f}", fill="black", font=fnt)
        ims.append(im.quantize(64))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps * 2), loop=0)


def draw(log, fps, out, tiles_idx, crop, inset, decision_word):
    s0, s1 = longest_run(log)
    w = np.array([e["pick"][2] if e["pick"] is not None else np.nan for e in log])
    h = np.array([e["pick"][3] if e["pick"] is not None else np.nan for e in log])
    fired, on_target = RR.decide(list(zip(w, h)), [e["on"] for e in log], fps, "full")
    rs = [band_test(w[s0:s1], fps), band_test(h[s0:s1], fps)]
    score = [(bool(r["fired"]), r["share"] / r["base"]) if r else (False, -1) for r in rs]
    col = 0 if score[0] >= score[1] else 1
    r = rs[col]; full = w if col == 0 else h; sig = full[s0:s1]
    assert bool(r["fired"]) == fired
    nrows_t = 2
    fig = plt.figure(figsize=(6.5, 11.6))
    outer = fig.add_gridspec(4, 1, height_ratios=[2.5, 0.42, 1.85, 1.0], hspace=0.32)
    gt_ = outer[0].subgridspec(2, 4, hspace=0.62, wspace=0.08)
    gb_ = outer[2].subgridspec(3, 1, height_ratios=[1.0, 0.13, 0.62], hspace=0.12)
    for t, i in enumerate(tiles_idx):
        ax = fig.add_subplot(gt_[t // 4, t % 4]); e = log[i]
        cx, cy = e["truth"]
        x0, y0 = int(cx - crop / 2), int(cy - crop / 2)
        H, W = e["fr"].shape[:2]
        tile = np.full((crop, crop, 3), 255, np.uint8)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + crop), min(H, y0 + crop)
        tile[ya - y0:yb - y0, xa - x0:xb - x0] = e["fr"][ya:yb, xa:xb]
        ax.imshow(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB), extent=(x0, x0 + crop, y0 + crop, y0), interpolation="nearest")
        if e["win"] is not None:
            ax.add_patch(Rectangle((e["win"][0] - 60, e["win"][1] - 60), 120, 120, fill=False, ec="white", ls="--", lw=1.2))
        if e["pred"] is not None:
            ax.plot(e["pred"][0], e["pred"][1], "+", color="cyan", ms=10, mew=1.6)
        for d in e["dets"]:
            ax.add_patch(Rectangle((d[0] - d[2] / 2, d[1] - d[3] / 2), d[2], d[3], fill=False, ec="yellow", lw=1.2))
        if e["pick"] is not None:
            p = e["pick"]
            ax.add_patch(Rectangle((p[0] - p[2] / 2 - 2, p[1] - p[3] / 2 - 2), p[2] + 4, p[3] + 4, fill=False, ec="lime", lw=1.6))
        if inset:
            ia = ax.inset_axes([0.0, 0.0, 0.4, 0.4])
            q = 2 * inset; sub = np.full((q, q, 3), 255, np.uint8)
            u0, v0 = int(cx - inset), int(cy - inset)
            ua, va, ub, vb = max(0, u0), max(0, v0), min(W, u0 + q), min(H, v0 + q)
            sub[va - v0:vb - v0, ua - u0:ub - u0] = e["fr"][va:vb, ua:ub]
            ia.imshow(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ia.set_xticks([]); ia.set_yticks([]); ia.grid(False)
            for sp in ia.spines.values():
                sp.set_edgecolor("white"); sp.set_linewidth(1.0)
        ax.set_xlim(x0, x0 + crop); ax.set_ylim(y0 + crop, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_xlabel(f"frame {i}\n{ST_NAME[e['state']]}\n$q_t$ = {e['conf']:.2f}", fontsize=14, labelpad=3, linespacing=1.0)
        if t == 0:
            ax.set_title("a", loc="left", fontsize=14)
    t = np.arange(len(log))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    RUN = "#FBE3D3"
    ax1 = fig.add_subplot(gb_[0])
    ax1.axvspan(s0, s1, color=RUN, lw=0, zorder=0)
    ax1.plot(t, [e["conf"] for e in log], color=ST.BLUE)
    ax1.axhline(0.25, color=ST.GREY, lw=1.0, ls="-.")
    ax1.set_ylim(0, 1); ax1.set_ylabel("$q_t$", labelpad=8); ax1.set_xlim(0, len(log))
    ax1.tick_params(labelbottom=False)
    ax1.set_title("b", loc="left", fontsize=14, pad=4)
    axl = fig.add_subplot(outer[1]); axl.axis("off")
    axl.legend([Line2D([], [], color=ST.BLUE), Line2D([], [], color=ST.GREY, ls="-.", lw=1.0),
                Patch(fc=RUN, ec="none"), Patch(fc=ST.GREEN, ec="none"), Patch(fc=ST.YELLOW, ec="none"),
                Line2D([], [], color=ST.ORANGE, marker="v", ls="none", ms=8)],
               ["$q_t$", "threshold 0.25", "longest run", "track maintenance", "extrapolation", "re-acquisition"],
               loc="center", bbox_to_anchor=(0.5, 0.35), ncol=3, borderaxespad=0)
    ax1s = fig.add_subplot(gb_[1], sharex=ax1)
    shades = {"hold": ST.GREEN, "reacq": ST.GREEN, "coast": ST.YELLOW, "lost": "white"}
    for i, e in enumerate(log):
        ax1s.add_patch(Rectangle((i, 0), 1, 1, color=shades[e["state"]], lw=0))
        if e["state"] == "reacq":
            ax1s.plot(i + 0.5, 0.5, "v", color=ST.ORANGE, ms=8, mec="black", mew=0.5)
    ax1s.set_ylim(0, 1); ax1s.set_yticks([]); ax1s.grid(False)
    ax1s.tick_params(labelbottom=False, bottom=False)
    ax2 = fig.add_subplot(gb_[2], sharex=ax1)
    ax2.axvspan(s0, s1, color=RUN, lw=0, zorder=0)
    ax2.plot(t, full, color=ST.ORANGE)
    ax2.set_ylabel(("$w_t$" if col == 0 else "$h_t$") + ", px", labelpad=8); ax2.set_xlabel("Frame number", labelpad=4)
    ax3 = fig.add_subplot(outer[3])
    x = sig - np.nanmean(sig); n = len(x); tt = np.arange(n)
    x = x - np.polyval(np.polyfit(tt, x, 2), tt)
    P = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(n, 1 / fps)
    ax3.axvspan(1.5, 6.0, color=RUN, lw=0, zorder=0)
    ax3.bar(f[1:], P[1:] / P[1:].sum(), width=f[1] * 0.8, color=ST.ORANGE, ec="none")
    ax3.set_xlabel("Frequency, Hz", labelpad=4); ax3.set_ylabel("Power\nshare", labelpad=8)
    ax3.set_title("c", loc="left", fontsize=14)
    verdict = "met" if r["fired"] else "not met"
    ax3.text(0.985, 0.95, f"$S$ = {r['share']:.2f};  $2S_w$ = {2 * r['base']:.2f};  $A$ = {r['amp']:.2f}\n"
             f"criterion {verdict}: {decision_word if r['fired'] else '“unknown”'}",
             transform=ax3.transAxes, ha="right", va="top", fontsize=14,
             bbox=dict(fc="white", ec="black", lw=0.6, pad=3))
    for a_ in (ax1, ax2, ax3):
        None
    fig.savefig(os.path.join(A4, "fig", out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    states = {s: sum(e["state"] == s for e in log) for s in ST_NAME}
    lowc = sum(e["conf"] < 0.25 for e in log[s0:s1])
    print(out, "primary rule: fired", fired, "on target", on_target, "frames", len(log), "run", s0, s1, "states", states, "col", "wh"[col], "share", round(r["share"], 3),
          "2Sw", round(2 * r["base"], 3), "amp", round(r["amp"], 3), "peak", round(r["peak"], 2), "fired", r["fired"],
          "run frames with c<0.25", lowc)
    return log


def bird(tid=17, seed=7, alp=False):
    tr = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}[tid]
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {int(p[0]): (p[1], p[2]) for p in tr["pts"]}
    ks = sorted(truth)
    holder = X.Holder(truth[ks[0]], "ext", period=int(X.FPS_MIDWAY), sigma=15.0, seed=seed, alp=alp)

    def it():
        for k, fr in frames(MIDWAY, ks[0], ks[-1]):
            if k in truth:
                tx, ty = truth[k]
                near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < X.GATE_DET]
                fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
                yield k, fr, (tx, ty), None if fl is None else (fl[0], fl[1]), det.get(k, [])
    log = run_holder(holder, it())
    cached = (load(os.path.join(CACHE, "e10_series.pkl"))[("bird", tid)][("att_ext1Hz", seed, True)][0] if alp
              else load(os.path.join(CACHE, "e3_series.pkl"))[(tid, "att_ext1Hz", seed)][0])
    print("bird: series identical to main run:", same(holder.series, cached))
    sfx = "_alp" if alp else ""
    draw(log, X.FPS_MIDWAY, f"en/fig_attn_bird{sfx}_en.png", [4, 18, 29, 50, 60, 70, 95, 115], 200, 0, "“bird”")
    pass  # no GIF for the English figure


def uav(seq=3, S=9, alp=False):
    i, idx = [s for s in D.segments() if s[0] == seq][0]
    fs, gt = D.load_seq(i)
    k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
    from ultralytics import YOLO
    from common import WEIGHTS
    model = YOLO(WEIGHTS)
    first = cv2.imread(fs[idx[0]])
    W, H = int(round(first.shape[1] / k)), int(round(first.shape[0] / k))
    imgsz = max(64, int(np.ceil(max(W, H) / 32.0) * 32))
    c0 = np.array([gt[idx[0]][0] + gt[idx[0]][2] / 2, gt[idx[0]][1] + gt[idx[0]][3] / 2]) / k
    holder = X.Holder(tuple(c0), "none", alp=alp)

    def it():
        for j in idx:
            fr = cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA)
            r = model.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0]
            dets = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                     float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                     float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
            yield j, fr, ((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k), None, dets
    log = run_holder(holder, it())
    cached = (load(os.path.join(CACHE, "e10_series.pkl"))[("uav", (i, idx[0], S))][("att_none", None, True)][0] if alp
              else load(os.path.join(CACHE, "e5_series.pkl"))[(i, idx[0], S)][("att_none", None)][0])
    print("uav: segment", i, idx[0], len(idx), "series identical to main run:", same(holder.series, cached))
    pick = list(np.linspace(5, len(log) - 5, 8).astype(int))
    sfx = "_alp" if alp else ""
    draw(log, D.FPS, f"en/fig_attn_uav{sfx}_en.png", pick, 140, 12, "“bird”")
    write_gif(log, 140, os.path.join(A4, "fig", "supp", f"attn_uav{sfx}.gif"), D.FPS)


if __name__ == "__main__":
    bird(alp=True)

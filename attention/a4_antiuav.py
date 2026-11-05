#!/usr/bin/env python3
"""E17. Holding under clouds: Anti-UAV RGB-T (Jiang et al., IEEE TMM 2023; 318 visible-band
videos, 1920x1080, 20 Hz, per-frame boxes). Post-hoc check asked by the author: does the
window threshold hold a small UAV against overcast / textured cloud sky, and does the primary
decision rule stay free of false "bird" decisions?

Differences from DUT (E5/E16), stated in the report:
  * a pan-tilt tracking camera keeps the UAV near the image centre, so the background (clouds,
    buildings) moves past the target, not the target across the sky;
  * a burnt-in HUD (black crosshair, white text) is removed per video: pixels that are constant
    over time and very dark or very bright are masked and inpainted (Telea, r = 3) before any
    processing;
  * night frames (mean grey < 60) are excluded.
Segment rule = E5/E16 (UAV darker by >= 8 grey levels; ring std <= 25 "open", > 25 "textured";
otherwise "notdark"; runs >= 96 frames). To bound run time ONE segment per video and group is
used (the longest run, first 300 frames). Equivalent sizes 6/9/12 px by frame reduction.

  python3 a4_antiuav.py --scan   -> cache/e17_segments.pkl, reports/e17_sheet_<group>.png
  python3 a4_antiuav.py          -> reports/e17_antiuav.csv, reports/e17_antiuav.txt
"""
import csv
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPO, REPORTS, WIN, load, local_blobs, save   # noqa: E402
import a4_drones as D                                                    # noqa: E402
import a4_experiments as X                                               # noqa: E402
import a4_multidet as M                                                  # noqa: E402
from a4_clutter import frame_stats, group_of                             # noqa: E402
from a4_multidet_table import run_on                                     # noqa: E402
from a4_rules import decide                                              # noqa: E402

AU = os.path.join(REPO, "antiuav", "data")
FPS = 20.0
NIGHT = 60.0


def videos():
    return sorted(glob.glob(os.path.join(AU, "*", "*", "visible.mp4")))


def gt_of(v):
    d = json.load(open(v.replace(".mp4", ".json")))
    return [r if (e and len(r) == 4 and r[2] > 0 and r[3] > 0) else None for e, r in zip(d["exist"], d["gt_rect"])]


def hud_mask(v):
    cap = cv2.VideoCapture(v); fr = []; j = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if j % 25 == 0:
            fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        j += 1
    A = np.array(fr, np.float32); sd = A.std(0); md = np.median(A, 0)
    m = ((sd < 4) & ((md < 50) | (md > 200))).astype(np.uint8)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def clean(f, mask):
    return cv2.inpaint(f, mask, 3, cv2.INPAINT_TELEA)


def clean_crop(f, mask, box):
    """Inpaint only the neighbourhood used by frame_stats (box + ring of one box size, + margin);
    returns a grey frame-sized array valid inside that neighbourhood and the offset-free box."""
    x, y, w, h = box; p = int(max(w, h)) * 2 + 16
    x0, y0 = max(0, int(x) - p), max(0, int(y) - p)
    x1, y1 = min(f.shape[1], int(x + w) + p), min(f.shape[0], int(y + h) + p)
    c = f[y0:y1, x0:x1]; m = mask[y0:y1, x0:x1]
    if m.any():
        c = cv2.inpaint(c, m, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (x - x0, y - y0, w, h)


def scan_one(v):
    cv2.setNumThreads(1)
    gt = gt_of(v); mask = hud_mask(v)
    cap = cv2.VideoCapture(v); lab = []
    for j in range(len(gt)):
        ok, f = cap.read()
        if not ok:
            break
        if gt[j] is None:
            lab.append(None); continue
        small = cv2.cvtColor(cv2.resize(f, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if small.mean() < NIGHT:
            lab.append(None); continue
        g, b = clean_crop(f, mask, gt[j])
        lab.append(group_of(frame_stats(g, b)))
    out = []
    for grp in ("open", "textured", "notdark"):
        best, run = [], []
        for j, x in enumerate(lab + [None]):
            if x == grp:
                run.append(j)
            else:
                if len(run) > len(best):
                    best = run
                run = []
        if len(best) >= D.SEG_MIN:
            out.append((grp, v, best[:D.SEG_MAX]))
    return out


def scan():
    from multiprocessing import Pool
    vs = videos(); segs = []
    with Pool(8) as pool:
        for n, r in enumerate(pool.imap(scan_one, vs)):
            segs += r
            if n % 20 == 0:
                print(f"  scanned {n + 1}/{len(vs)} videos, {len(segs)} segments", flush=True)
    save(segs, os.path.join(CACHE, "e17_segments.pkl"))
    for grp in ("open", "textured", "notdark"):
        g = [s for s in segs if s[0] == grp]
        print(f"{grp}: {len(g)} segments, {sum(len(p) for _, _, p in g)} frames")
        tiles = []
        for _, v, idx in g:
            gt = gt_of(v); j = idx[len(idx) // 2]
            cap = cv2.VideoCapture(v); cap.set(cv2.CAP_PROP_POS_FRAMES, j); ok, f = cap.read()
            if not ok:
                continue
            x, y, w, h = gt[j]; cx, cy, r = int(x + w / 2), int(y + h / 2), int(max(w, h) * 4)
            c = f[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            c = cv2.resize(c, (160, 160))
            cv2.putText(c, f"{len(tiles)}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            tiles.append(c)
        if tiles:
            cols = 12
            rows = [np.hstack(tiles[k:k + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[k:k + cols])))
                    for k in range(0, len(tiles), cols)]
            cv2.imwrite(os.path.join(REPORTS, f"e17_sheet_{grp}.png"), np.vstack(rows))


def run_segment(model, v, idx, S, mask):
    gt = gt_of(v)
    s0 = float(np.median([max(gt[j][2], gt[j][3]) for j in idx]))
    k = s0 / S
    if k < 1.0:
        return None, None
    cap = cv2.VideoCapture(v); cap.set(cv2.CAP_PROP_POS_FRAMES, idx[0])
    W = H = None; holders = None; conf, vis, on = [], [], {}
    for j in range(idx[0], idx[-1] + 1):
        ok, f = cap.read()
        if not ok:
            break
        if j not in idx:
            continue
        f = clean(f, mask)
        if W is None:
            W, H = int(round(f.shape[1] / k)), int(round(f.shape[0] / k))
            imgsz = max(64, int(math.ceil(max(W, H) / 32.0) * 32))
            x, y, w, h = gt[j]; c0 = (float(x + w / 2) / k, float(y + h / 2) / k)
            holders = [("tbd", None, X.TBD(c0)), ("att_none", None, X.Holder(c0, "none"))]
            for sd in D.SEEDS:
                holders.append(("att_ext1Hz", sd, X.Holder(c0, "ext", period=int(FPS), sigma=15.0, seed=sd)))
        fr = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        x, y, w, h = gt[j]; tx, ty = (x + w / 2) / k, (y + h / 2) / k
        r = model.predict(fr, conf=0.05, imgsz=imgsz, device="cpu", verbose=False)[0]
        dets = [(float((b.xyxy[0][0] + b.xyxy[0][2]) / 2), float((b.xyxy[0][1] + b.xyxy[0][3]) / 2),
                 float(b.xyxy[0][2] - b.xyxy[0][0]), float(b.xyxy[0][3] - b.xyxy[0][1]),
                 float(b.conf[0]), int(b.cls[0])) for b in r.boxes]
        near = [d for d in dets if np.hypot(d[0] - tx, d[1] - ty) < D.GATE_DET]
        best = max(near, key=lambda d: d[4], default=None)
        conf.append(best[4] if best else 0.0)
        vis.append(any(np.hypot(b[0] - tx, b[1] - ty) < D.ON_T for b in local_blobs(fr, tx, ty)))
        for mode, sd, obj in holders:
            if mode == "tbd":
                obj.step(dets)
            else:
                obj.step(fr, (tx, ty), None, dets)
            ps = obj.pos[-1]
            # on-target gate: the article's 25 px and a tight gate of 2 target sizes
            d_ = None if ps is None else float(np.hypot(ps[0] - tx, ps[1] - ty))
            on.setdefault((mode, sd), []).append(d_)
    rows = []
    for mode, sd, obj in holders:
        dist = on[(mode, sd)]
        o25 = [d is not None and d < D.ON_T for d in dist]
        o2s = [d is not None and d < 2 * S for d in dist]
        s = list(obj.series)
        f_full, _ = decide(s, o25, FPS, "full"); f_any, _ = decide(s, o25, FPS, "any")
        rows.append(dict(video=os.path.relpath(v, AU), f0=idx[0], n=len(idx), size_px=S, k=round(k, 2),
                         cov_025=float(np.mean([c >= 0.25 for c in conf])), vis=float(np.mean(vis)),
                         mode=mode, seed=-1 if sd is None else sd, on25=float(np.mean(o25)), on2s=float(np.mean(o2s)),
                         strict25=bool(run_on(s, o25)), strict2s=bool(run_on(s, o2s)),
                         bird_full=bool(f_full), bird_any=bool(f_any)))
    return rows, conf


_MODEL = None


def _init():
    global _MODEL
    import torch
    torch.set_num_threads(1); cv2.setNumThreads(1)
    from ultralytics import YOLO
    _MODEL = YOLO(D.WEIGHTS)


def run_one(seg):
    grp, v, idx = seg
    mask = hud_mask(v); out = []
    for S in D.SIZES:
        r, _ = run_segment(_MODEL, v, idx, S, mask)
        if r:
            for x in r:
                x["group"] = grp
            out += r
    return out


def main():
    if "--scan" in sys.argv:
        scan(); return
    segs = load(os.path.join(CACHE, "e17_segments.pkl"))
    keep = None
    if os.path.exists(os.path.join(REPORTS, "e17_keep.json")):
        keep = json.load(open(os.path.join(REPORTS, "e17_keep.json")))   # visual tags per group
    from multiprocessing import Pool
    rows = []
    with Pool(8, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(run_one, segs)):
            rows += r
            print(f"  {n + 1}/{len(segs)} done", flush=True)
            if n % 20 == 0 and rows:
                with open(os.path.join(REPORTS, "e17_antiuav.csv"), "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(REPORTS, "e17_antiuav.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary(rows, keep)


def summary(rows, keep=None):
    L = ["E17 Anti-UAV RGB (tracking camera, HUD inpainted, day only; post-hoc)", ""]
    for grp in ("open", "textured", "notdark"):
        g = [r for r in rows if r["group"] == grp]
        if not g:
            continue
        nseg = len({(r["video"], r["f0"]) for r in g})
        L.append(f"{grp}: {nseg} segments")
        for S in D.SIZES:
            rs = [r for r in g if r["size_px"] == S]
            if not rs:
                continue
            u0 = [r for r in rs if r["mode"] == "att_none"]; e1 = [r for r in rs if r["mode"] == "att_ext1Hz"]
            tb = [r for r in rs if r["mode"] == "tbd"]

            def pc(a, key):
                k = sum(r[key] for r in a); lo, hi = M.wilson(k, len(a))
                return f"{k}/{len(a)} ({100 * k / len(a):.0f} %, {lo:.0f}-{hi:.0f})"
            L.append(f"  {S:2d} px (n={len(u0)}): cov@0.25 median {100 * np.median([r['cov_025'] for r in u0]):.0f} % | "
                     f"vis median {100 * np.median([r['vis'] for r in u0]):.0f} % | strict 25px TBD {pc(tb, 'strict25')} "
                     f"U0 {pc(u0, 'strict25')} UZ-1 {pc(e1, 'strict25')} | strict 2S U0 {pc(u0, 'strict2s')} UZ-1 {pc(e1, 'strict2s')} | "
                     f"false bird full U0 {sum(r['bird_full'] for r in u0)}/{len(u0)} UZ-1 {sum(r['bird_full'] for r in e1)}/{len(e1)} "
                     f"TBD {sum(r['bird_full'] for r in tb)}/{len(tb)} | any U0 {sum(r['bird_any'] for r in u0)}/{len(u0)} "
                     f"UZ-1 {sum(r['bird_any'] for r in e1)}/{len(e1)}")
        L.append("")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e17_antiuav.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

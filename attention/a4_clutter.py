#!/usr/bin/env python3
"""E16. Holding on a cluttered background: DUT Anti-UAV segments REJECTED by the E5
open-sky rule (post-hoc check asked by the author: clouds / textured sky).

E5 kept frames where the UAV is darker than its background ring by >= 8 grey levels AND
the ring is uniform (std <= 25). Here the complement is split into two groups:
  textured -- UAV darker by >= 8, ring std > 25 (textured background: cloud edges,
              broken clouds, structures);
  notdark  -- UAV NOT darker by 8 (same brightness or lighter than the background:
              the method looks only for dark silhouettes, formula (2)).
Same segment rule (contiguous runs >= 96 frames, split at 300), same equivalent sizes
6/9/12 px, same holders (TBD, attention window without cue, external cue 1 Hz x 5 seeds)
and the same metrics as the article: visibility, strict held run (>= 48 frames, >= 50 %
on target), false "bird" decisions by the primary rule (one test on the longest run)
and by the "any window" rule.

Step 1 (--scan): list segments + contact sheet of background crops (reports/e16_sheet.png).
Step 2 (default): run holders. Outputs: reports/e16_clutter.csv, reports/e16_clutter.txt
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, WIN, save          # noqa: E402
import a4_drones as D                                 # noqa: E402
from a4_multidet_table import run_on                  # noqa: E402
from a4_rules import decide                           # noqa: E402
import a4_multidet as M                               # noqa: E402

FPS = 25.0


def frame_stats(g, box):
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
    obj = g[y0:y1, x0:x1]
    p = int(max(w, h))
    ring = g[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p].astype(float).copy()
    ring[min(p, y0):min(p, y0) + (y1 - y0), min(p, x0):min(p, x0) + (x1 - x0)] = np.nan
    if obj.size == 0 or np.isnan(ring).all():
        return None
    return float(np.percentile(obj, 20)) - float(np.nanmedian(ring)), float(np.nanstd(ring))


def group_of(st):
    if st is None:
        return None
    dark, sd = st
    if dark <= D.DARK and sd <= D.RING_STD:
        return "open"           # E5 open-sky frames (already in the article)
    if dark <= D.DARK:
        return "textured"
    return "notdark"


def segments():
    out = []
    for i in range(1, 21):
        fs, gt = D.load_seq(i)
        lab = []
        for f, b in zip(fs, gt):
            lab.append(group_of(frame_stats(cv2.imread(f, cv2.IMREAD_GRAYSCALE), b)))
        for grp in ("textured", "notdark"):
            run = []
            for j, v in enumerate(lab + [None]):
                if v == grp:
                    run.append(j)
                else:
                    if len(run) >= D.SEG_MIN:
                        for s in range(0, len(run), D.SEG_MAX):
                            part = run[s:s + D.SEG_MAX]
                            if len(part) >= D.SEG_MIN:
                                out.append((grp, i, part))
                    run = []
    return out


def sheet(segs):
    tiles = []
    for grp, i, idx in segs:
        fs, gt = D.load_seq(i)
        j = idx[len(idx) // 2]
        img = cv2.imread(fs[j]); x, y, w, h = gt[j]
        cx, cy, r = int(x + w / 2), int(y + h / 2), int(max(w, h) * 4)
        crop = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        crop = cv2.resize(crop, (200, 200))
        cv2.putText(crop, f"{grp[:4]} v{i:02d} f{idx[0]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(crop)
    cols = 8
    rows = [np.hstack(tiles[k:k + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[k:k + cols])))
            for k in range(0, len(tiles), cols)]
    cv2.imwrite(os.path.join(REPORTS, "e16_sheet.png"), np.vstack(rows))


def main():
    segs = segments()
    for grp in ("textured", "notdark"):
        g = [s for s in segs if s[0] == grp]
        print(f"{grp}: {len(g)} segments, {sum(len(p) for _, _, p in g)} frames, sequences {sorted({i for _, i, _ in g})}")
    save(segs, os.path.join(CACHE, "e16_segments.pkl"))
    if segs:
        sheet(segs)
    if "--scan" in sys.argv:
        return
    from ultralytics import YOLO
    model = YOLO(D.WEIGHTS)
    rows = []
    for n, (grp, i, idx) in enumerate(segs):
        for S in D.SIZES:
            D.SERIES.clear()
            r = D.run_segment(model, i, idx, S)
            if not r:
                continue
            ser = D.SERIES[(i, idx[0], S)]
            for row in r:
                key = (row["mode"], None if row["seed"] == -1 else row["seed"])
                s, o = ser[key]
                f_full, _ = decide(s, o, FPS, "full")
                f_any, _ = decide(s, o, FPS, "any")
                rows.append(dict(row, group=grp, strict=bool(run_on(s, o)), bird_full=bool(f_full), bird_any=bool(f_any)))
        print(f"  {n + 1}/{len(segs)} {grp} video{i:02d} ({len(idx)} frames) done", flush=True)
    with open(os.path.join(REPORTS, "e16_clutter.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["E16 cluttered background (DUT segments rejected by the open-sky rule; post-hoc)", ""]
    for grp in ("textured", "notdark"):
        g = [s for s in segs if s[0] == grp]
        L.append(f"{grp}: {len(g)} segments, {sum(len(p) for _, _, p in g)} frames, sequences {sorted({i for _, i, _ in g})}")
        for S in D.SIZES:
            rs = [r for r in rows if r["group"] == grp and r["size_px"] == S]
            if not rs:
                continue
            u0 = [r for r in rs if r["mode"] == "att_none"]
            e1 = [r for r in rs if r["mode"] == "att_ext1Hz"]
            tb = [r for r in rs if r["mode"] == "tbd"]

            def pc(a, key):
                k = sum(r[key] for r in a); lo, hi = M.wilson(k, len(a))
                return f"{k}/{len(a)} ({100 * k / len(a):.0f} %, {lo:.0f}-{hi:.0f})"
            L.append(f"  {S:2d} px: vis median {100 * np.median([r['vis'] for r in u0]):.0f} % (min {100 * min(r['vis'] for r in u0):.0f} %) | "
                     f"strict TBD {pc(tb, 'strict')} U0 {pc(u0, 'strict')} UZ-1 {pc(e1, 'strict')} | "
                     f"false bird full: U0 {sum(r['bird_full'] for r in u0)}/{len(u0)} UZ-1 {sum(r['bird_full'] for r in e1)}/{len(e1)} "
                     f"TBD {sum(r['bird_full'] for r in tb)}/{len(tb)} | any: U0 {sum(r['bird_any'] for r in u0)}/{len(u0)} "
                     f"UZ-1 {sum(r['bird_any'] for r in e1)}/{len(e1)}")
        L.append("")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e16_clutter.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

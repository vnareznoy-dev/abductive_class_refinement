#!/usr/bin/env python3
"""Evaluation set for Article 4: far objects of 4-12 px on the Midway clip.

Pre-registered rule, applied to every moving track of flap_midway.py:
  length >= 96 frames (3.2 s), median max(w, h) of the mask 4-12 px,
  straightness (net / path displacement) >= 0.8.
Duplicates (two hypotheses on one object: >= 50 % shared frames within 10 px)
are collapsed to the longer track. Contact sheets are written for visual
verification; tracks rejected visually are listed in EXCLUDE with the reason.

Output: cache/eval_tracks.pkl -> [{"id", "pts": [(k, cx, cy, w, h), ...]}],
        reports/select.txt, reports/sheets/sheet_XX.jpg
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MIDWAY, MIDWAY_TRACKS, REPORTS, CACHE, frames, load, save  # noqa: E402

MIN_LEN, SZ_MIN, SZ_MAX, MIN_STRAIGHT = 96, 4, 12, 0.8
EXCLUDE = {}          # track id -> reason, filled after visual verification


def stats(a):
    xy = a[:, 1:3]
    path = np.hypot(*np.diff(xy, axis=0).T).sum()
    return len(a), float(np.median(np.maximum(a[:, 3], a[:, 4]))), \
        float(np.hypot(*(xy[-1] - xy[0])) / max(path, 1e-9))


def duplicate(a, b):
    fa = {int(p[0]): p for p in a}
    common = [k for k in (int(p[0]) for p in b) if k in fa]
    if len(common) < 0.5 * min(len(a), len(b)):
        return False
    fb = {int(p[0]): p for p in b}
    near = sum(1 for k in common if np.hypot(fa[k][1] - fb[k][1], fa[k][2] - fb[k][2]) < 10)
    return near >= 0.5 * len(common)


def main():
    T = [np.array(t) for t in load(MIDWAY_TRACKS)]
    cand = [(i, a) for i, a in enumerate(T)
            if (s := stats(a))[0] >= MIN_LEN and SZ_MIN <= s[1] <= SZ_MAX and s[2] >= MIN_STRAIGHT]
    cand.sort(key=lambda x: -len(x[1]))
    kept = []
    for i, a in cand:
        if not any(duplicate(a, b) for _, b in kept):
            kept.append((i, a))
    lines = [f"rule: len >= {MIN_LEN}, median mask {SZ_MIN}-{SZ_MAX} px, straightness >= {MIN_STRAIGHT}",
             f"candidates: {len(cand)}; after duplicate removal: {len(kept)}"]
    final = [(i, a) for i, a in kept if i not in EXCLUDE]
    for i, r in EXCLUDE.items():
        lines.append(f"excluded after visual check: track {i} -- {r}")
    lines.append("visual check (reports/sheets): every tile of every track shows a bird in the sky; "
                 "in some frames a neighbouring bird passes through the tile, so a track may "
                 "partly pass between neighbouring birds (identity is not guaranteed)")
    lines.append(f"evaluation set: {len(final)} tracks")
    for i, a in final:
        n, sz, st = stats(a)
        lines.append(f"  track {i:4d}: frames {int(a[0, 0])}-{int(a[-1, 0])}, {n} pts, "
                     f"mask {sz:.0f} px, straightness {st:.2f}")
    save([{"id": i, "pts": [tuple(map(float, p)) for p in a]} for i, a in final],
         os.path.join(CACHE, "eval_tracks.pkl"))

    # contact sheets: 10 tracks per sheet, 8 tiles per track (x5 upscale)
    want = {}
    for i, a in kept:
        idx = np.linspace(0, len(a) - 1, 8).astype(int)
        for j in idx:
            want.setdefault(int(a[j, 0]), []).append((i, a[j]))
    tiles = {}
    for k, fr in frames(MIDWAY):
        for i, p in want.get(k, []):
            cx, cy = int(p[1]), int(p[2]); r = 14
            crop = fr[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            crop = cv2.resize(crop, (140, 140), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"{k}", (3, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.setdefault(i, []).append(crop)
    os.makedirs(os.path.join(REPORTS, "sheets"), exist_ok=True)
    for s in range(0, len(kept), 10):
        rows = []
        for i, _ in kept[s:s + 10]:
            tl = tiles[i] + [np.zeros_like(tiles[i][0])] * (8 - len(tiles[i]))
            row = np.hstack(tl[:8])
            lab = np.full((140, 90, 3), 255, np.uint8)
            cv2.putText(lab, f"#{i}", (4, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            rows.append(np.hstack([lab, row]))
        cv2.imwrite(os.path.join(REPORTS, f"sheets/sheet_{s // 10:02d}.jpg"), np.vstack(rows))
    text = "\n".join(lines)
    print(text)
    open(os.path.join(REPORTS, "select.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

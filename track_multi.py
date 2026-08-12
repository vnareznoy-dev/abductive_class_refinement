#!/usr/bin/env python3
"""Multi-hypothesis tracking for the silhouette channel.

The binding constraint on the silhouette channel turned out to be DATA
ASSOCIATION, not the spectral criterion (see the article): three separate runs
locked onto a wave crest, a twig tip, or broke up entirely. A single greedy
"most contrasted blob" track has no way to prefer the bird.

The discriminator used here is physical rather than photometric: a flying bird
TRANSLATES, while a wind-shaken twig oscillates about a fixed point and a wave
crest appears and vanishes. So maintain SEVERAL track hypotheses at once and
keep the one whose net displacement is large relative to the path it walked:

    straightness = |end - start| / sum(|step|)

near 1 for a bird crossing the frame, near 0 for anything jittering in place.
This is the same "zeroth-order quantities are robust" argument as everywhere
else in this work: net displacement is a DC quantity, path length is not, and
their ratio suppresses per-frame noise.
"""
from __future__ import annotations

import sys
import cv2
import numpy as np


def dark_blobs(frame, mask_top=0.0, mask_bottom=1.0, pct=99.85,
               min_area=6, max_area=2500, max_w=70, max_h=50):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    if mask_top > 0:
        g[:int(mask_top * H), :] = 255
    if mask_bottom < 1.0:
        g[int(mask_bottom * H):, :] = 255
    dark = cv2.GaussianBlur(g, (0, 0), 7) - g
    thr = max(float(np.percentile(dark, pct)), 8.0)
    m = (dark > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        bx, by, bw, bh = cv2.boundingRect(c)
        if (min_area < bw * bh < max_area and 2 <= bw <= max_w
                and 2 <= bh <= max_h):
            out.append((bx + bw / 2.0, by + bh / 2.0, float(bw), float(bh),
                        float(cv2.contourArea(c))))
    return out


class Track:
    __slots__ = ("pts", "vel", "miss", "alive")

    def __init__(self, k, c):
        self.pts = [(k,) + c]
        self.vel = np.zeros(2)
        self.miss = 0
        self.alive = True

    def predict(self):
        return np.array(self.pts[-1][1:3]) + self.vel

    def add(self, k, c):
        prev = np.array(self.pts[-1][1:3])
        self.vel = 0.6 * self.vel + 0.4 * (np.array(c[:2]) - prev)
        self.pts.append((k,) + c)
        self.miss = 0

    def straightness(self):
        a = np.array([(p[1], p[2]) for p in self.pts], float)
        if len(a) < 3:
            return 0.0
        path = float(np.hypot(*np.diff(a, axis=0).T).sum())
        net = float(np.hypot(*(a[-1] - a[0])))
        return net / path if path > 1e-6 else 0.0


def track_multi(frames, gate=30.0, coast=5, min_len=48, **blob_kw):
    """Maintain many hypotheses; return runs sorted by straightness x length."""
    tracks, done = [], []
    for k, fr in enumerate(frames):
        cands = dark_blobs(fr, **blob_kw)
        taken = set()
        for t in tracks:
            if not t.alive:
                continue
            pred = t.predict()
            best, bj = gate, None
            for j, c in enumerate(cands):
                if j in taken:
                    continue
                d = float(np.hypot(c[0] - pred[0], c[1] - pred[1]))
                if d < best:
                    best, bj = d, j
            if bj is None:
                t.miss += 1
                if t.miss > coast:
                    t.alive = False
                    if len(t.pts) >= min_len:
                        done.append(t)
            else:
                taken.add(bj)
                t.add(k, cands[bj])
        # unclaimed candidates start new hypotheses
        for j, c in enumerate(cands):
            if j not in taken:
                tracks.append(Track(k, c))
        tracks = [t for t in tracks if t.alive]
        if len(tracks) > 400:                      # keep the longest hypotheses
            tracks.sort(key=lambda t: -len(t.pts))
            tracks = tracks[:400]
    for t in tracks:
        if len(t.pts) >= min_len:
            done.append(t)
    done.sort(key=lambda t: -(t.straightness() * len(t.pts)))
    return done


def read_all(path, limit=4000):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = []
    while len(out) < limit:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out, fps


if __name__ == "__main__":
    from flap_real import band_test
    path = sys.argv[1]
    mt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    mb = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    frames, fps = read_all(path)
    print(f"{len(frames)} кадрів, {fps:.0f} Гц, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    ts = track_multi(frames, mask_top=mt, mask_bottom=mb)
    print(f"гіпотез >= 48 кадрів: {len(ts)}")
    for i, t in enumerate(ts[:4]):
        a = np.array([(p[0], p[1], p[2], p[3], p[4]) for p in t.pts], float)
        w, h = a[:, 3], a[:, 4]
        print()
        print(f"--- гіпотеза {i+1}: кадри {int(a[0,0])}..{int(a[-1,0])} "
              f"({len(a)} = {len(a)/fps:.1f} с), рамка {w.mean():.1f}x{h.mean():.1f} px, "
              f"прямолінійність {t.straightness():.2f} ---")
        for lab, sig in (("розмах w", w), ("висота h", h)):
            r = band_test(sig, fps)
            if r:
                print(f"    {lab:10s} частка {r['share']:.2f} "
                      f"({r['share']/r['base']:.1f}x) пік {r['peak']:.2f} Гц "
                      f"ампл {100*r['amp']:5.1f}% -> "
                      f"{'МАХИ' if r['fired'] else 'нема'}")
        tiles = []
        for p in t.pts[4:16]:
            k, cx, cy, bw, bh, _ = p
            fr = frames[int(k)]
            pad = 24
            x0, x1 = int(max(0, cx - bw/2 - pad)), int(min(fr.shape[1], cx + bw/2 + pad))
            y0, y1 = int(max(0, cy - bh/2 - pad)), int(min(fr.shape[0], cy + bh/2 + pad))
            cr = cv2.resize(fr[y0:y1, x0:x1], (130, 130),
                            interpolation=cv2.INTER_NEAREST)
            cv2.putText(cr, f"{int(k)}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
            tiles.append(cr)
        if len(tiles) >= 12:
            import os
            nm = os.path.splitext(os.path.basename(path))[0][:14]
            out = f"vid2/frames/multi_{nm}_h{i+1}.jpg"
            cv2.imwrite(out, np.vstack([np.hstack(tiles[0:6]),
                                        np.hstack(tiles[6:12])]))
            print(f"    -> {out}")

#!/usr/bin/env python3
"""Article 3, version v1.1 (revision for the Scientific Bulletin of UNFU): recount of the silhouette channel
with independent units.

R1 specificity: unit = continuous UAV run of the birdrone set (data_derived/uav_runs.csv). The criterion (12) is checked
   ONCE on the whole run, for each of the four measures (w, h, area, h/w); a run fires if any measure fires.
   Runs shorter than 48 frames are reported separately (the article requires n >= 48). Also: the rule "any 48-frame
   window" (step 1) to show the inflation of a multiple check; frame rate 25 Hz (assumed) and 30 Hz.
R2 sensitivity: (a) the three published bird tracks (data_derived/bird_tracks.csv); Midway track 338 has gaps;
   (b) the 50 Midway bird tracks of the multiple-hypothesis tracker (data_derived/midway50_tracks.csv) selected by a fixed rule
   (>= 96 frames, mask 4-12 px, straightness >= 0.8; visually checked on contact sheets), single check on the longest
   gap-free run of >= 48 frames, 29.97 Hz.
R3 sensitivity of the hand-set constants of (12): multiplier 1.5-3, amplitude 0.01-0.05, band edges, n.
Wilson 95 % intervals for every share.
Run: python3 rev_flap.py   (needs only data_derived/). Output: reports/r1_r3_flap.txt"""
import csv
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, "data_derived")
OUT = os.path.join(HERE, "reports", "r1_r3_flap.txt")
MIN_RUN = 48


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100*k/n:.0f} %, 95 % CI {100*lo:.0f}-{100*hi:.0f} %)" if n else "0/0"


def band_test(sig, fps, lo=1.5, hi=6.0, mult=2.0, amp_min=0.02, nmin=MIN_RUN):
    """criterion (12) exactly as flap_real.band_test / flap_specificity.band_test, parameters exposed"""
    x = np.asarray(sig, float)
    n = len(x)
    if n < nmin:
        return None
    mean = float(np.mean(x))
    if abs(mean) < 1e-9:
        return None
    x = x - mean
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 2), t)
    if np.allclose(x, 0):
        return None
    p = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (fr >= lo) & (fr <= hi)
    tot = p[1:].sum()
    if tot <= 0 or not band.any():
        return None
    share = float(p[band].sum() / tot)
    base = (hi - lo) / (fps / 2.0)
    amp = float(np.sqrt(2.0 * p[band].sum()) / n / abs(mean))
    return dict(share=share, base=base, ratio=share / base, amp=amp, peak=float(fr[band][np.argmax(p[band])]),
                fired=share > mult * base and amp > amp_min)


def measures(w, h):
    w = np.asarray(w, float); h = np.asarray(h, float)
    return {"w": w, "h": h, "area": w * h, "h/w": h / np.maximum(w, 1e-6)}


def fires(w, h, fps, ms=("w", "h", "area", "h/w"), **kw):
    M = measures(w, h)
    res = {m: band_test(M[m], fps, **kw) for m in ms}
    return any(r is not None and r["fired"] for r in res.values()), res


def any_window(w, h, fps, n=MIN_RUN, **kw):
    """the multiple rule: fires if ANY n-frame window (step 1) fires on any measure"""
    for s in range(0, len(w) - n + 1):
        if fires(w[s:s + n], h[s:s + n], fps, **kw)[0]:
            return True
    return False


def load_csv(path):
    T = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            T[r["track"]].append((int(float(r["frame"])), float(r["w"]), float(r["h"])))
    return {k: sorted(v) for k, v in T.items()}


def longest_gapfree(rows):
    best, cur = [], [rows[0]]
    for a, b in zip(rows, rows[1:]):
        if b[0] - a[0] == 1:
            cur.append(b)
        else:
            best = max(best, cur, key=len); cur = [b]
    return max(best, cur, key=len)


def midway50():
    """the 50 tracks in the order of the file; the key is the track number of the multiple-hypothesis tracker"""
    T = {}
    with open(os.path.join(DD, "midway50_tracks.csv")) as f:
        for r in csv.DictReader(f):
            T.setdefault(int(r["track"].split("_")[1]), []).append((int(float(r["frame"])), float(r["w"]), float(r["h"])))
    return {k: sorted(v) for k, v in T.items()}


def main():
    L = []
    P = lambda s="": (L.append(s), print(s))
    uav = load_csv(os.path.join(DD, "uav_runs.csv"))
    birds = load_csv(os.path.join(DD, "bird_tracks.csv"))
    mid = midway50()
    P("R1. Specificity of criterion (12) on the birdrone UAV runs (manual annotation boxes)")
    for fps in (25.0, 30.0):
        long_ = [k for k in uav if len(uav[k]) >= MIN_RUN]
        short = [k for k in uav if len(uav[k]) < MIN_RUN]
        once = sum(fires([r[1] for r in uav[k]], [r[2] for r in uav[k]], fps)[0] for k in long_)
        anyw = sum(any_window([r[1] for r in uav[k]], [r[2] for r in uav[k]], fps) for k in long_)
        nmin_short = sum(fires([r[1] for r in uav[k]], [r[2] for r in uav[k]], fps, nmin=32)[0] for k in short)
        P(f"  {fps:.0f} Hz: runs >= 48 frames {len(long_)} ({sum(len(uav[k]) for k in long_)} frames); fired once-per-run: {ci(once, len(long_))}; "
          f"rule 'any 48-frame window': {ci(anyw, len(long_))}; runs < 48 frames {len(short)} (checked with n >= 32: fired {nmin_short})")
    for k in sorted(uav, key=lambda k: int(k.split('_')[1])):
        ok, res = fires([r[1] for r in uav[k]], [r[2] for r in uav[k]], 25.0, nmin=32)
        P(f"    {k}: {len(uav[k])} frames; ratio S/S0 " + ", ".join(f"{m} {res[m]['ratio']:.2f}" for m in res if res[m]) + f"; fired {ok}")
    P()
    P("R2a. The three published bird tracks")
    fps_b = {"cormorant": 25.0, "midway_338": 29.97}
    for k, rows in birds.items():
        fps = fps_b.get(k, 30.0)
        fr = [r[0] for r in rows]
        gaps = [b for a, b in zip(fr, fr[1:]) if b - a > 1]
        seg = longest_gapfree(rows)
        ok, res = fires([r[1] for r in seg], [r[2] for r in seg], fps)
        best = max((r for r in res.values() if r), key=lambda r: r["ratio"], default=None)
        P(f"  {k}: frames {fr[0]}-{fr[-1]}, {len(rows)} points, gaps before frames {gaps}; longest gap-free run {len(seg)} frames; "
          f"single check fired {ok}" + (f" (best S/S0 {best['ratio']:.2f}, peak {best['peak']:.2f} Hz)" if best else " (n < 48: criterion not applicable)"))
        if gaps:
            full = {r[0]: r for r in rows}
            ks = range(fr[0], fr[-1] + 1)
            wi = np.interp(list(ks), fr, [r[1] for r in rows]); hi_ = np.interp(list(ks), fr, [r[2] for r in rows])
            ok2, res2 = fires(wi, hi_, fps)
            b2 = max((r for r in res2.values() if r), key=lambda r: r["ratio"])
            P(f"     with the {len(list(ks)) - len(rows)} missing frames interpolated ({len(list(ks))} frames): fired {ok2}, best S/S0 {b2['ratio']:.2f}, peak {b2['peak']:.2f} Hz "
              + ", ".join(f"{m} {res2[m]['ratio']:.2f}{'*' if res2[m]['fired'] else ''}" for m in res2 if res2[m]))
    P()
    P("R2b. 50 Midway bird tracks (fixed selection rule, visually checked), single check on the longest gap-free run, 29.97 Hz")
    n_app = n_fire = 0; seglens = []; fired_ids = []
    for tid, rows in mid.items():
        seg = longest_gapfree(rows); seglens.append(len(seg))
        if len(seg) < MIN_RUN:
            continue
        n_app += 1
        ok, _ = fires([r[1] for r in seg], [r[2] for r in seg], 29.97)
        n_fire += ok; fired_ids += [tid] if ok else []
    P(f"  longest gap-free run: median {np.median(seglens):.0f} frames (min {min(seglens)}, max {max(seglens)}); criterion applicable (>= 48) on {n_app} of 50")
    P(f"  fired: {ci(n_fire, n_app)} of applicable tracks; tracks {fired_ids}")
    anyw = sum(any_window([r[1] for r in longest_gapfree(rows)], [r[2] for r in longest_gapfree(rows)], 29.97) for rows in mid.values() if len(longest_gapfree(rows)) >= MIN_RUN)
    P(f"  rule 'any 48-frame window' on the same runs: {ci(anyw, n_app)}")
    sizes = [float(np.median([max(r[1], r[2]) for r in rows])) for rows in mid.values()]
    P(f"  median silhouette size of the 50 tracks: {np.median(sizes):.1f} px (range {min(sizes):.0f}-{max(sizes):.0f})")
    P()
    P("R3. Sensitivity to the hand-set constants of (12): UAV runs >= 48 fired (25 Hz / 30 Hz) | published birds fired | Midway 50 fired")
    variants = [("published: mult 2, amp 0.02, band 1.5-6, n 48", {}),
                ("mult 1.5", dict(mult=1.5)), ("mult 2.5", dict(mult=2.5)), ("mult 3", dict(mult=3.0)),
                ("amp 0.01", dict(amp_min=0.01)), ("amp 0.05", dict(amp_min=0.05)),
                ("band 1.0-6", dict(lo=1.0)), ("band 2.0-6", dict(lo=2.0)), ("band 1.5-5", dict(hi=5.0)), ("band 1.5-8", dict(hi=8.0)),
                ("n 32", dict(nmin=32)), ("n 64", dict(nmin=64))]
    long_ = [k for k in uav if len(uav[k]) >= MIN_RUN]
    for name, kw in variants:
        u25 = sum(fires([r[1] for r in uav[k]], [r[2] for r in uav[k]], 25.0, **kw)[0] for k in long_)
        u30 = sum(fires([r[1] for r in uav[k]], [r[2] for r in uav[k]], 30.0, **kw)[0] for k in long_)
        b = 0
        for k, rows in birds.items():
            seg = rows if k == "midway_338" else longest_gapfree(rows)
            if k == "midway_338":
                fr = [r[0] for r in rows]; ks = list(range(fr[0], fr[-1] + 1))
                seg = list(zip(ks, np.interp(ks, fr, [r[1] for r in rows]), np.interp(ks, fr, [r[2] for r in rows])))
            b += fires([r[1] for r in seg], [r[2] for r in seg], fps_b.get(k, 30.0), **kw)[0]
        m = sum(fires([r[1] for r in longest_gapfree(rows)], [r[2] for r in longest_gapfree(rows)], 29.97, **kw)[0]
                for rows in mid.values() if len(longest_gapfree(rows)) >= kw.get("nmin", MIN_RUN))
        P(f"  {name:48s} UAV {u25}/{len(long_)} / {u30}/{len(long_)} | birds {b}/3 | Midway {m}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

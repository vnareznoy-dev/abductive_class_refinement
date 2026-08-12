#!/usr/bin/env python3
"""SPECIFICITY of the silhouette flap detector: does it fire on a NON-flapping object?

A flap detector is worthless if it also fires on drones, and specificity is the
half of the claim that CAN be tested on the data at hand. The birdrone drone
frames are nearly continuous -- 9 runs of >= 32 frames, 2738 frames in total,
the longest 1104 (44 s at 25 fps) -- whereas the bird frames are decimated to
runs of 6. So:

  * specificity (no firing on drones)  -> testable NOW, done here;
  * sensitivity (firing on birds)      -> needs continuous single-bird video.

Criterion, as in silhouette.py: the share of spectral power inside the wing-beat
band must exceed twice the white-noise expectation for that band, AND the
in-band amplitude must be physically meaningful. White noise alone already puts
(hi-lo)/f_Nyquist of its power in the band, so a threshold below that share
would fire on noise by construction.

CAVEAT, stated rather than hidden: these are HAND-ANNOTATED boxes, so the
sequence carries annotation noise which need not be white -- annotator drift is
low-frequency, which if anything makes this test optimistic in the band of
interest. It measures the channel as instrumented by boxes; a real system would
use a segmentation mask.
"""
from __future__ import annotations

import os, re, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rf/birdrone")
FPS = 25.0                    # assumed; stated as an assumption
BEAT_LO, BEAT_HI = 1.5, 6.0
MIN_RUN = 32
FLAP_RATIO_MULT = 2.0         # must exceed 2x the white-noise share
MIN_AMP = 0.02                # in-band amplitude, relative to the mean


def collect_drone():
    got = {}
    for sub in ("train", "valid"):
        for im in glob.glob(os.path.join(SRC, sub, "images", "*")):
            b = os.path.basename(im)
            stem = b.split("_jpg")[0] if "_jpg" in b else b.split(".rf.")[0]
            if not re.match(r"^\d+$", stem):
                continue
            lb = os.path.join(SRC, sub, "labels",
                              os.path.splitext(b)[0] + ".txt")
            if not os.path.exists(lb):
                continue
            g = [l.split() for l in open(lb) if l.strip()]
            g = [x for x in g if len(x) >= 5]
            if len(g) != 1 or int(g[0][0]) != 1:
                continue
            n = int(stem)
            got.setdefault(n, (float(g[0][3]) * 640.0, float(g[0][4]) * 640.0))
    ks = sorted(got)
    runs, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b - a == 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    return [np.array([got[k] for k in r]) for r in runs if len(r) >= MIN_RUN]


def band_test(sig, fps=FPS, lo=BEAT_LO, hi=BEAT_HI):
    x = np.asarray(sig, float)
    n = len(x)
    if n < MIN_RUN:
        return None
    mean = float(np.mean(x))
    if abs(mean) < 1e-9:
        return None
    x = x - x.mean()
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 2), t)      # remove slow drift
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
    peak = float(fr[band][np.argmax(p[band])])
    fired = share > FLAP_RATIO_MULT * base and amp > MIN_AMP
    return dict(share=share, base=base, amp=amp, peak=peak, n=n, fired=fired)


def main():
    runs = collect_drone()
    print(f"непереривних дронових прогонів >= {MIN_RUN} кадрів: {len(runs)}; "
          f"кадрів разом: {sum(len(r) for r in runs)}")
    print(f"частота кадрів прийнята {FPS:.0f} Гц; полоса махів "
          f"{BEAT_LO}-{BEAT_HI} Гц; біла полка = "
          f"{(BEAT_HI-BEAT_LO)/(FPS/2):.2f}; критерій = частка > "
          f"{FLAP_RATIO_MULT:.0f}x полки ТА амплітуда > {100*MIN_AMP:.0f} %")
    print()
    hdr = (f"{'прогін':>7s} {'кадрів':>7s} {'рамка,px':>9s} | " +
           " | ".join(f"{s:>26s}" for s in ("висота", "площа", "h/w")))
    print(hdr); print("-" * len(hdr))
    fires = 0
    total = 0
    for i, a in enumerate(runs):
        w, h = a[:, 0], a[:, 1]
        area = w * h
        ar = h / np.maximum(w, 1e-6)
        cells = []
        for sig in (h, area, ar):
            r = band_test(sig)
            total += 1
            if r is None:
                cells.append("—".rjust(26)); continue
            if r["fired"]:
                fires += 1
            cells.append(f"частка {r['share']:.2f} ампл {100*r['amp']:4.1f}% "
                         f"{'МАХИ' if r['fired'] else 'нема'}".rjust(26))
        print(f"{i+1:7d} {len(a):7d} {np.sqrt(w*h).mean():9.0f} | " +
              " | ".join(cells))

    print()
    print("=" * 96)
    print(f"СПЕЦИФІЧНІСТЬ: детектор махів спрацював на {fires} з {total} "
          f"сигналів нерухомокрилого об'єкта")
    if fires == 0:
        print("хибних спрацьовувань немає -> критерій НЕ приймає дрон за птаха.")
        print("це половина твердження; чутливість (спрацьовування НА птаху)")
        print("вимагає непереривного відео одиночного птаха і НЕ перевірена.")
    else:
        print("є хибні спрацьовування -> критерій у поточному вигляді")
        print("непридатний: він приймав би дрон за птаха.")


if __name__ == "__main__":
    main()

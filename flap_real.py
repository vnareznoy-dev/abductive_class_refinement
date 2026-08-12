#!/usr/bin/env python3
"""THE SILHOUETTE CHANNEL ON REAL FOOTAGE: distant flapping bird -> ALP says bird.

Footage: "Kormoran (Tiefflug)", Mathe-AC Lernwerkstatt Aachen, CC BY 4.0,
Wikimedia Commons. A great cormorant in low flight over the river Ems, filmed
from behind at long range: the bounding box is 7-15 px wide, i.e. exactly the
regime where the detector's appearance evidence is weakest.

WHY MOG2 FAILS HERE, AND WHAT REPLACES IT. Background subtraction locks onto
rippling water and onto the burned-in subtitle text, both of which move far more
than a 12-px bird. The bird is instead DARK against light water, so it is found
as a compact negative-contrast blob relative to a local blurred background. The
subtitle band is masked out explicitly.

WHY THE ASPECT RATIO IS THE SIGNAL. Seen from behind, a wing beat mostly changes
the SPAN: wings level -> wide and flat, wings raised -> narrow and tall. So h/w
carries the beat more strongly than h alone.

The criterion is the same one whose SPECIFICITY was verified on 2738 continuous
non-flapping drone frames with zero false fires (flap_specificity.py): in-band
power share must exceed twice the white-noise expectation, and the in-band
amplitude must be physically meaningful. The criterion is deliberately blind to
amplitude scale and sensitive to spectral SHAPE -- which is what lets it work
even though box-level noise amplitude (38 %) exceeds the beat amplitude.
"""
from __future__ import annotations

import os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VID = os.path.join(HERE, "vid2/Kormoran_Tiefflug.webm")
BEAT_LO, BEAT_HI = 1.5, 6.0
MIN_RUN = 48
RATIO_MULT = 2.0
MIN_AMP = 0.02


def dark_blobs(frame):
    """Compact negative-contrast regions, subtitle band and sky band removed."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    g[int(0.82 * H):, :] = 255            # burned-in subtitles
    g[:int(0.05 * H), :] = 255
    dark = cv2.GaussianBlur(g, (0, 0), 9) - g
    thr = max(float(np.percentile(dark, 99.9)), 8.0)
    m = (dark > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        bx, by, bw, bh = cv2.boundingRect(c)
        if 20 < bw * bh < 6000 and 4 <= bw <= 140 and 3 <= bh <= 90:
            out.append((bx + bw / 2.0, by + bh / 2.0, float(bw), float(bh),
                        float(cv2.contourArea(c))))
    return out


def track(path, f0, f1, gate=45.0, coast=5):
    """Gated nearest-neighbour track over a frame range."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    seq, last, vel, miss = [], None, np.zeros(2), 0
    for k in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        cands = dark_blobs(fr)
        if last is None:
            pick = max(cands, key=lambda c: c[4]) if cands else None
        else:
            pred = np.array(last[:2]) + vel
            pick, best = None, gate
            for c in cands:
                d = float(np.hypot(c[0] - pred[0], c[1] - pred[1]))
                if d < best:
                    best, pick = d, c
        if pick is None:
            seq.append(None); miss += 1
            if miss > coast:
                last, vel, miss = None, np.zeros(2), 0
            continue
        if last is not None:
            vel = 0.6 * vel + 0.4 * (np.array(pick[:2]) - np.array(last[:2]))
        last, miss = pick, 0
        seq.append((k,) + pick)
    cap.release()
    return seq, fps


def runs(seq, min_len=MIN_RUN):
    out, cur = [], []
    for s in seq:
        if s is None:
            if len(cur) >= min_len:
                out.append(cur)
            cur = []
        else:
            cur.append(s)
    if len(cur) >= min_len:
        out.append(cur)
    return out


def band_test(sig, fps):
    x = np.asarray(sig, float)
    n = len(x)
    mean = float(np.mean(x))
    if n < MIN_RUN or abs(mean) < 1e-9:
        return None
    x = x - x.mean()
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 2), t)          # remove slow drift
    if np.allclose(x, 0):
        return None
    p = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (fr >= BEAT_LO) & (fr <= BEAT_HI)
    tot = p[1:].sum()
    if tot <= 0 or not band.any():
        return None
    share = float(p[band].sum() / tot)
    base = (BEAT_HI - BEAT_LO) / (fps / 2.0)
    amp = float(np.sqrt(2.0 * p[band].sum()) / n / abs(mean))
    peak = float(fr[band][np.argmax(p[band])])
    return dict(share=share, base=base, amp=amp, peak=peak, n=n,
                fired=(share > RATIO_MULT * base and amp > MIN_AMP))


def main():
    f0, f1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (1240, 1800)
    if not os.path.exists(VID):
        sys.exit(f"немає відео: {VID}")
    print(f"відео: {os.path.basename(VID)}  кадри {f0}..{f1}")
    seq, fps = track(VID, f0, f1)
    rs = runs(seq)
    hit = sum(1 for s in seq if s is not None)
    print(f"частота {fps:.0f} Гц; детекцій {hit}/{len(seq)}; "
          f"прогонів >= {MIN_RUN} кадрів: {len(rs)}")
    print(f"полоса {BEAT_LO}-{BEAT_HI} Гц; біла полка "
          f"{(BEAT_HI-BEAT_LO)/(fps/2):.2f}; критерій частка > "
          f"{RATIO_MULT:.0f}x полки ТА ампл > {100*MIN_AMP:.0f} %")
    if not rs:
        sys.exit("немає достатньо довгого прогону")

    any_fire = False
    for i, r in enumerate(rs):
        a = np.array([(s[1], s[2], s[3], s[4], s[5]) for s in r], float)
        w, h, area = a[:, 2], a[:, 3], a[:, 4]
        ar = h / np.maximum(w, 1e-6)
        print()
        print(f"--- прогін {i+1}: кадри {r[0][0]}..{r[-1][0]} "
              f"({len(r)} кадрів = {len(r)/fps:.1f} с), "
              f"рамка {w.mean():.1f}x{h.mean():.1f} px ---")
        for label, sig in (("розмах w", w), ("висота h", h),
                           ("площа", area), ("відношення h/w", ar)):
            t = band_test(sig, fps)
            if t is None:
                print(f"    {label:16s} —")
                continue
            if t["fired"]:
                any_fire = True
            print(f"    {label:16s} частка {t['share']:.2f} (полка {t['base']:.2f}, "
                  f"{t['share']/t['base']:.1f}x)  пік {t['peak']:.2f} Гц  "
                  f"ампл {100*t['amp']:5.1f} %  -> "
                  f"{'МАХИ' if t['fired'] else 'нема'}")

    # ---- feed the result into the abductive theory ---------------------
    print()
    print("=" * 84)
    print("ПОДАЄМО РЕЗУЛЬТАТ У АБДУКТИВНУ ТЕОРІЮ")
    print("=" * 84)
    from pipeline import Abducer
    abd = Abducer(os.path.join(HERE, "air_objects.pl"))
    # the appearance channel is deliberately WRONG and confident -- the case the
    # method exists for: a distant 12-px object called a drone by the detector
    f = {"f_flap": any_fire, "f_hover": 0.0, "f_speed": None,
         "f_speed_cv": None, "f_turn_irreg": None, "f_straight": None,
         "f_turn_bias": None}
    cls, why = abd.explain(f, "drone", 0.95)
    print(f"детектор каже: drone (упевненість 0.95)")
    print(f"канал силуету дає: f_flap({'true' if any_fire else 'false'})")
    print(f"кінематичні факти: не заземлені (об'єкт 12 px на дальності)")
    print()
    print(f"ВИСНОВОК ALP: {cls}")
    print(f"ОБҐРУНТУВАННЯ:  {why}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sensitivity of the abductive theory to its authored thresholds.

WHY THIS EXISTS. Five thresholds in `air_objects.pl` are authored domain
knowledge rather than fitted values: hover_frac, turn_irreg, straight_frac,
speed_cv_bird, circling_bias. The obvious objection is "they were tuned on the
test set". Tuning them here would CONFIRM that objection, so this script does
the opposite: it sweeps each one and reports how wide the PLATEAU of equivalent
performance is. The claim under test is not "our value is optimal" but "the
conclusion does not depend on the exact value".

READING A FLAT RESULT CORRECTLY -- the lesson of the first attempt. The first
version of this script reported 0.0 pp spread on four thresholds out of five and
that looked like excellent robustness. It was an artefact: the swept ranges did
not intersect the observed feature distributions at all (f_turn_irreg ranged
36-41 rad/s against a threshold swept over 0.10-2.00). A sweep that misses the
data cannot change a single decision, so it says nothing whatever about
robustness. Therefore every sweep here is printed TOGETHER WITH:

  * the feature's measured support per class (p5..p95), so it is visible
    whether the swept range even reaches the data;
  * the share of tracks on which the feature is GROUNDABLE at all -- after the
    groundability tests T1-T3 (see the article), derivative features are withheld
    on most tracks at realistic sigma, and a threshold on a fact that is never
    asserted is vacuous rather than robust.

A flat curve therefore has three possible meanings, and only the first is a
result: (1) genuinely insensitive, sweep covers the data; (2) sweep misses the
data; (3) the feature is not groundable, so no rule using it ever fires.
"""
from __future__ import annotations

import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev                                   # noqa: E402
from pipeline import Abducer                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DET_ERR = 0.20        # detector error rate

# threshold -> (authored value, the feature it tests, sweep values)
SWEEPS = {
    "hover_frac":    (0.15, "f_hover",
                      [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.85]),
    "turn_irreg":    (0.35, "f_turn_irreg",
                      [0.05, 0.10, 0.20, 0.35, 0.60, 1.00, 3.00, 10.0]),
    "straight_frac": (0.70, "f_straight",
                      [0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]),
    "speed_cv_bird": (0.25, "f_speed_cv",
                      [0.02, 0.05, 0.10, 0.25, 0.40, 0.60, 1.00, 2.00]),
    "circling_bias": (0.10, "f_turn_bias",
                      [0.005, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]),
}


def prepare(items, sigma, det_err=DET_ERR, seed=3):
    """Features and detector labels computed ONCE; only the theory then varies."""
    rng = np.random.default_rng(seed)
    rows = [(lbl, ev.featurise(pos, sigma, rng)) for lbl, pos, _tag in items]
    drng = np.random.default_rng(11)
    out = []
    for lbl, f in rows:
        wrong = drng.random() < det_err
        det = ("bird" if lbl == "drone" else "drone") if wrong else lbl
        out.append((lbl, f, det))
    return out


def strict_acc(prepared, abd):
    hit = 0
    for lbl, f, det in prepared:
        cls, _why = abd.explain(f, det, 0.9)
        hit += int(cls == lbl)
    return hit / len(prepared)


def plateau(values, accs, tol=0.01):
    """Longest contiguous run of values whose accuracy is within `tol` of best."""
    best = max(accs)
    good = [a >= best - tol for a in accs]
    bi = bj = ci = None
    span = 0
    for k, g in enumerate(good + [False]):
        if g and ci is None:
            ci = k
        elif not g and ci is not None:
            if k - ci > span:
                span, bi, bj = k - ci, ci, k - 1
            ci = None
    return (values[bi], values[bj], best) if bi is not None else (None, None, best)


def support(prep, key):
    """Measured support of a feature per class + groundable share."""
    got = {}
    for cls in ("drone", "bird"):
        v = [f[key] for lbl, f, _d in prep
             if lbl == cls and f.get(key) is not None]
        got[cls] = (np.percentile(np.array(v, float), [5, 95]) if v else None,
                    len(v), sum(1 for lbl, _f, _d in prep if lbl == cls))
    return got


def run(name, items, abd, sigma):
    prep = prepare(items, sigma)
    print()
    print("=" * 98)
    print(f"{name}   (sigma = {sigma} м, помилка детектора {DET_ERR*100:.0f} %, "
          f"{len(prep)} треків)")
    print("=" * 98)
    base = strict_acc(prep, abd)
    print(f"авторська конфігурація: строга точність {base:.3f}")

    for thr_name, (authored, feat_key, values) in SWEEPS.items():
        accs = []
        for v in values:
            abd.set_thr(thr_name, v)
            accs.append(strict_acc(prep, abd))
        abd.set_thr(thr_name, authored)                # restore
        lo, hi, best = plateau(values, accs)
        sup = support(prep, feat_key)

        print()
        print(f"--- {thr_name}  (авторське {authored:g}; ознака {feat_key}) ---")
        for cls in ("drone", "bird"):
            q, n_ok, n_tot = sup[cls]
            frac = 100.0 * n_ok / max(1, n_tot)
            if q is None:
                print(f"    носій {cls:>5s}: ознака НЕ ЗАЗЕМЛЕНА ні на одному "
                      f"треку (0 / {n_tot})")
            else:
                print(f"    носій {cls:>5s}: p5..p95 = {q[0]:.3f}..{q[1]:.3f}   "
                      f"заземлено на {n_ok}/{n_tot} = {frac:.0f} % треків")
        cells = " ".join((f"[{v:g}:{a:.3f}]" if v == authored else f"{v:g}:{a:.3f}")
                         for v, a in zip(values, accs))
        print(f"    свип: {cells}")
        spread = max(accs) - min(accs)

        # classify the meaning of a flat curve
        any_ground = any(sup[c][1] > 0 for c in ("drone", "bird"))
        reach = False
        for c in ("drone", "bird"):
            q = sup[c][0]
            if q is not None and min(values) <= q[1] and max(values) >= q[0]:
                reach = True
        # Strongest possible robustness: the two class supports are DISJOINT and
        # every swept value falls in the gap between them. Then no threshold in
        # that gap can decide differently -- the threshold is unconstrained, not
        # merely insensitive. This must not be reported as a failed sweep.
        qd, qb = sup["drone"][0], sup["bird"][0]
        disjoint_gap = False
        if qd is not None and qb is not None:
            lo_g, hi_g = (qb[1], qd[0]) if qb[1] < qd[0] else (qd[1], qb[0])
            if lo_g < hi_g and all(lo_g <= v <= hi_g for v in values):
                disjoint_gap = (lo_g, hi_g)

        if spread < 0.005:
            if not any_ground:
                verdict = "ВАКУУМ: ознака не заземлена, правило не спрацьовує ніколи"
            elif disjoint_gap:
                verdict = ("НЕ ОБМЕЖЕНИЙ: носії класів роз'єднані, будь-яке "
                           f"значення у зазорі {disjoint_gap[0]:.3f}.."
                           f"{disjoint_gap[1]:.3f} дає той самий результат")
            elif not reach:
                verdict = "НЕВАЛІДНО: діапазон свипа не перетинає дані"
            else:
                verdict = "СТІЙКО: свип покриває дані, рішення не змінюються"
        else:
            inside = "авторське У плато" if (lo is not None and lo <= authored <= hi) \
                     else "авторське ПОЗА плато"
            verdict = (f"плато {lo:g}..{hi:g} (best {best:.3f}), {inside}")
        print(f"    розкид {spread*100:5.1f} п.п.  ->  {verdict}")


def main():
    abd = Abducer(os.path.join(HERE, "air_objects.pl"))
    general = ev.build_set()
    inreg = ev.hovering_drones(60) + [i for i in general if i[0] == "bird"]
    # sigma = 0 shows the theory as designed; sigma = 0.30 shows it as deployed
    for sg in (0.0, 0.30):
        run("РЕЖИМ, У ЯКОМУ ОБМЕЖЕННЯ АКТИВНЕ (дрони утримують позицію)",
            inreg, abd, sg)
    print()
    print("=" * 98)
    print("трактування: плоска крива — результат ЛИШЕ тоді, коли свип покриває")
    print("носій ознаки й ознака заземлена. Інакше це вакуум або хибний свип.")


if __name__ == "__main__":
    main()

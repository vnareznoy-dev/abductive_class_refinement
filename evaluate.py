#!/usr/bin/env python3
"""Head-to-head evaluation of the abductive class-revision method.

Test set (identical observation pipeline for every class and every method):
  drones : 100 REAL CTU-MRS tracks, 25 Hz
  birds  :  90 SYNTHESISED tracks (30 per behaviour mode) -- declared synthetic,
           because open bird data at >=10 Hz does not exist (see birds.py)

Methods compared:
  1. detector-only      -- what the neural detector claims, no symbolic layer
  2. threshold rule     -- a plain non-abductive rule over the same features
                           (isolates the contribution of the abductive structure)
  3. trained classifier -- logistic regression on the same features + detector
                           label (a learned counterpart in the spirit of the
                           trajectory-LSTM baselines in the literature)
  4. ABDUCTIVE (ours)   -- air_objects.pl

Sweeps:
  A. detector error rate 0..40 % at fixed grounding error  -> does the symbolic
     layer actually repair detector mistakes?
  B. grounding error sigma 0..1.0 m at fixed detector error -> how far does the
     repair survive degraded grounding?
"""
from __future__ import annotations

import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv                                             # noqa: E402
import birds, features as feat                        # noqa: E402


def load_trajectory(path, stride=1):
    """(t[N], pos[N,3], vel[N,3]) from a resampled_trajectory_data_*.csv
    of the CTU-MRS set (columns: time, pos_x/y/z, vel_x/y/z)."""
    ts, ps, vs = [], [], []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % stride:
                continue
            ts.append(float(row["time"]))
            ps.append([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])])
            vs.append([float(row["vel_x"]), float(row["vel_y"]), float(row["vel_z"])])
    return np.array(ts), np.array(ps), np.array(vs)
from pipeline import Abducer, hover_fraction          # noqa: E402

DT = 0.04
WIN = 200                     # 8 s per decision
CTU = os.path.join(os.environ.get("CTU_MRS_DIR", "data/ctu_mrs"),
       "data/csv/resampled_trajectory_data_*.csv")


# ---------------------------------------------------------------- test set
def build_set():
    items = []
    for f in sorted(glob.glob(CTU)):
        _t, pos, _v = load_trajectory(f, stride=2)
        if len(pos) >= WIN:
            items.append(("drone", pos[:WIN], "real"))
    for tr in birds.make_bird_set(30, WIN, seed=17):
        items.append(("bird", tr["pos"], tr["mode"]))
    return items


def hovering_drones(n, seed=41, drift=0.15):
    """Station-keeping drones -- the regime the hover constraint is FOR.

    CTU-MRS contains evasive pursuit-evasion targets that essentially never
    hold position (hover fraction 0.03), so the strongest integrity constraint
    can never fire on it. Station keeping is, however, the archetypal
    drone/bird confusion case in surveillance, and the regime in which this
    method claims to act -- so it is evaluated separately and declared as such.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pos = np.zeros((WIN, 3)); p = np.array([0., 0., 60.]); v = np.zeros(3)
        for k in range(WIN):
            v = 0.9 * v + rng.normal(0, drift, 3) * DT
            p = p + v * DT
            pos[k] = p
        out.append(("drone", pos, "hover"))
    return out


def featurise(pos, sigma, rng):
    f = feat.extract(pos, DT, sigma, rng)
    f["f_hover"] = hover_fraction(pos, DT)      # robust DC estimator
    return f


# ---------------------------------------------------------------- methods
def m_detector(f, det, _conf):
    return det


def m_threshold(f, det, _conf):
    """Plain rule, no abduction: fixed precedence, no consistency check."""
    if f["f_hover"] > 0.15:
        return "drone"
    if f["f_flap"]:
        return "bird"
    if f["f_speed"] > 25.0:
        return "drone"
    return det


class Trained:
    """Logistic regression on the same features + detector label."""
    KEYS = ("f_speed", "f_speed_cv", "f_hover", "f_flap",
            "f_turn_irreg", "f_straight", "f_turn_bias")

    def __init__(self):
        self.w = None

    @staticmethod
    def _x(f, det):
        # A learned model has no way to express "not measurable": it must be
        # handed a number for every input, so an ungroundable feature is imputed
        # with 0. This is exactly the representational gap the abductive layer
        # does not have -- there, an unstated fact simply activates no rule.
        v = [(0.0 if f.get(k) is None else float(f[k])) for k in Trained.KEYS]
        v.append(1.0 if det == "drone" else 0.0)
        v.append(1.0)
        return np.array(v)

    def fit(self, X, y, iters=800, lr=0.05):
        X = np.asarray(X); y = np.asarray(y, float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        self.mu, self.sd = mu, sd
        Z = (X - mu) / sd
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Z @ w))
            w -= lr * (Z.T @ (p - y)) / len(y)
        self.w = w

    def __call__(self, f, det, _conf):
        z = (self._x(f, det) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-z @ self.w))
        return "drone" if p >= 0.5 else "bird"


def make_abductive(abd):
    def run(f, det, conf):
        cls, _why = abd.explain(f, det, conf)
        return cls
    return run


# ---------------------------------------------------------------- scoring
def score(pred, truth):
    pred = np.array(pred); truth = np.array(truth)
    decided = pred != "unknown"
    acc = float((pred[decided] == truth[decided]).mean()) if decided.any() else 0.0
    return dict(acc=acc,
                cover=float(decided.mean()),
                # accuracy over ALL items, an abstention counting as wrong
                strict=float((pred == truth).mean()))


def run_sweep(items, sigmas, err_rates, abd, seed=3):
    """Precompute features once per (item, sigma); then apply every method."""
    cache = {}
    for sg in sigmas:
        rng = np.random.default_rng(seed)
        cache[sg] = [(lbl, featurise(pos, sg, rng), tag) for lbl, pos, tag in items]

    # train the learned baseline on HALF the items at the lowest sigma
    tr_sg = sigmas[0]
    rows = cache[tr_sg]
    idx = np.arange(len(rows)); np.random.default_rng(0).shuffle(idx)
    tr_idx = idx[: len(idx) // 2]
    trained = Trained()
    X = [Trained._x(rows[i][1], rows[i][0]) for i in tr_idx]
    y = [1.0 if rows[i][0] == "drone" else 0.0 for i in tr_idx]
    trained.fit(X, y)

    methods = [("детектор", m_detector), ("правило", m_threshold),
               ("навчений", trained), ("АБДУКЦІЯ", make_abductive(abd))]

    out = []
    for sg in sigmas:
        for er in err_rates:
            rng = np.random.default_rng(11)
            det_labels, truths, fs = [], [], []
            for lbl, f, _tag in cache[sg]:
                wrong = rng.random() < er
                det = ("bird" if lbl == "drone" else "drone") if wrong else lbl
                det_labels.append(det); truths.append(lbl); fs.append(f)
            row = {"sigma": sg, "err": er}
            for name, fn in methods:
                preds = [fn(f, d, 0.9) for f, d in zip(fs, det_labels)]
                row[name] = score(preds, truths)
            out.append(row)
    return out


def main():
    items = build_set()
    n_d = sum(1 for l, _, _ in items if l == "drone")
    print(f"тестовий набір: {n_d} дронів (РЕАЛЬНІ CTU-MRS) + "
          f"{len(items)-n_d} птахів (синтетичні), вікно {WIN} відсчётів = {WIN*DT:.0f} с\n")
    abd = Abducer(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "air_objects.pl"))

    print("=" * 96)
    print("А. ЗРОСТАННЯ ПОМИЛКИ ДЕТЕКТОРА (sigma = 0.30 м)")
    print("=" * 96)
    res = run_sweep(items, [0.30], [0.0, 0.1, 0.2, 0.3, 0.4], abd)
    hdr = f"{'помилка дет.':>13s} | " + " | ".join(f"{n:>19s}" for n in
          ("детектор", "правило", "навчений", "АБДУКЦІЯ"))
    print(hdr); print("-" * len(hdr))
    print(f"{'':13s} | " + " | ".join(f"{'точн  покр  строг':>19s}" for _ in range(4)))
    for r in res:
        cells = []
        for n in ("детектор", "правило", "навчений", "АБДУКЦІЯ"):
            s = r[n]
            cells.append(f"{s['acc']:.2f} {s['cover']:.2f} {s['strict']:.2f}".rjust(19))
        print(f"{r['err']*100:11.0f} % | " + " | ".join(cells))

    print()
    print("=" * 96)
    print("Б. ЗРОСТАННЯ ПОХИБКИ ЗАЗЕМЛЕННЯ (помилка детектора = 20 %)")
    print("=" * 96)
    res = run_sweep(items, [0.0, 0.15, 0.30, 0.60, 1.0], [0.2], abd)
    print(hdr); print("-" * len(hdr))
    print(f"{'':13s} | " + " | ".join(f"{'точн  покр  строг':>19s}" for _ in range(4)))
    for r in res:
        cells = []
        for n in ("детектор", "правило", "навчений", "АБДУКЦІЯ"):
            s = r[n]
            cells.append(f"{s['acc']:.2f} {s['cover']:.2f} {s['strict']:.2f}".rjust(19))
        print(f"sigma={r['sigma']:6.2f} м | " + " | ".join(cells))
    print("\nточн = точність серед ВИРІШЕНИХ, покр = частка вирішених, "
          "строг = точність по всіх (воздержання = помилка)")

    # ---- IN-REGIME: station-keeping drones vs birds -----------------------
    items2 = hovering_drones(60) + [i for i in items if i[0] == "bird"]
    print()
    print("=" * 96)
    print("В. РЕЖИМ, У ЯКОМУ ОБМЕЖЕННЯ АКТИВНЕ: дрони УТРИМУЮТЬ ПОЗИЦІЮ vs птахи")
    print("   (60 висящих дронів + 90 птахів; помилка детектора 20 %)")
    print("=" * 96)
    res = run_sweep(items2, [0.0, 0.30, 0.60, 1.0], [0.2], abd)
    print(hdr); print("-" * len(hdr))
    print(f"{'':13s} | " + " | ".join(f"{'точн  покр  строг':>19s}" for _ in range(4)))
    for r in res:
        cells = []
        for n in ("детектор", "правило", "навчений", "АБДУКЦІЯ"):
            s2 = r[n]
            cells.append(f"{s2['acc']:.2f} {s2['cover']:.2f} {s2['strict']:.2f}".rjust(19))
        print(f"sigma={r['sigma']:6.2f} м | " + " | ".join(cells))


if __name__ == "__main__":
    main()

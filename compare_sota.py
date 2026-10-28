#!/usr/bin/env python3
"""Head-to-head against the two closest published optical approaches, on OUR data.

Competitors reimplemented as faithful feature-level proxies (the originals are
not open source; both are trained classifiers over engineered features):

  A. "TF-features" -- Drones (MDPI) 10(2):98, Jan 2026: kinematic features
     (CV of velocity) + time-frequency features of the BOUNDING-BOX AREA
     sequence (their TF-SVE entropy proxied by spectral entropy / flatness) +
     trajectory complexity (Higuchi fractal dimension), stacked classifier
     (proxied by logistic regression -- our compute budget, declared).
  B. "Trajectory-only" -- drone-vs-bird flight-trajectory characterisation
     (Coluccia et al. line): CV of speed, straightness, turn statistics of the
     centroid track, learned classifier.
  C. OURS -- fixed physical criterion on the silhouette series (band share >
     2x white-noise floor AND amplitude > 2 %), NO training, plus principled
     abstention.

Data: real sequences only.
  birds : the two visually verified tracks (cormorant 48 f @25 Hz;
          heron/stork 89 f @30 Hz, camera-motion compensated)
  drones: the 9 continuous hand-annotated birdrone runs (2738 f) --
          single-object frames, numeric stems, as in flap_specificity.py

Windows of 48 frames (stride 24) per track. FOUR protocols:

  P1  leave-one-TRACK-out CV -- the standard optimistic protocol;
  P2  CROSS-DOMAIN: train on cormorant + drone runs 1-5, test on heron +
      drone runs 6-9. The bird the classifier is tested on comes from a video,
      species, background and frame rate it has never seen. Our method takes
      no training, so P2 is where the comparison is meaningful.
  P3  BOX-NOISE STRESS: models trained on clean P2 training windows, tested on
      the P2 test windows with Gaussian noise (sigma px) on cx, cy, w, h.
  P4  DECOYS: false detector tracks on water (alp_on_failures.py, truth: no
      object) fed to the same models; requires reports/decoy_tracks.pkl.

HONESTY BOX (also printed): bird sample is TINY -- 2 independent tracks, so
bird-side rates carry Wilson intervals and windows within a track are
correlated. The learned proxies are OUR implementations of the cited ideas,
not the authors' code; conclusions are about the architectures (trained
features vs fixed physical criterion), not about the specific papers' scores.
"""
from __future__ import annotations

import os, re, sys, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flap_real as FR                                   # noqa: E402

WIN, STRIDE = 48, 24
P3_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
P3_SEEDS = (7, 11, 23, 42, 77)


# ------------------------------------------------------------------ data
def bird_tracks():
    """[(name, fps, [(cx,cy,w,h)...])] -- the two visually verified tracks."""
    out = []
    seq, fps = FR.track(FR.VID, 1240, 1800)
    ref = FR.runs(seq)[0]
    out.append(("bird:cormorant", fps,
                [(p[1], p[2], p[3], p[4]) for p in ref]))

    # heron: camera-motion-compensated multi-hypothesis track (as in 11.3)
    from track_multi import read_all, dark_blobs, Track
    frames, fps2 = read_all(os.path.join(HERE,
                            "vid2/Kamenitý_rybník_letící_pták.ogv"))
    cum = [np.zeros(2)]
    prev = np.float32(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))
    for k in range(1, len(frames)):
        cur = np.float32(cv2.cvtColor(frames[k], cv2.COLOR_BGR2GRAY))
        s, _ = cv2.phaseCorrelate(prev, cur)
        prev = cur
        cum.append(cum[-1] + np.array(s))
    tracks, done = [], []
    for k, fr in enumerate(frames):
        cw = [(c[0]-cum[k][0], c[1]-cum[k][1], c[2], c[3], c[4])
              for c in dark_blobs(fr, mask_bottom=0.50)]
        taken = set()
        for t in tracks:
            if not t.alive:
                continue
            pred = t.predict()
            best, bj = 30.0, None
            for j, c in enumerate(cw):
                if j in taken:
                    continue
                d = float(np.hypot(c[0]-pred[0], c[1]-pred[1]))
                if d < best:
                    best, bj = d, j
            if bj is None:
                t.miss += 1
                if t.miss > 5:
                    t.alive = False
                    if len(t.pts) >= WIN:
                        done.append(t)
            else:
                taken.add(bj); t.add(k, cw[bj])
        for j, c in enumerate(cw):
            if j not in taken:
                tracks.append(Track(k, c))
        tracks = [t for t in tracks if t.alive]
        if len(tracks) > 400:
            tracks.sort(key=lambda t: -len(t.pts)); tracks = tracks[:400]
    done += [t for t in tracks if len(t.pts) >= WIN]
    cand = [t for t in done if int(t.pts[0][0]) <= 2 and len(t.pts) >= 80]
    t = max(cand, key=lambda x: len(x.pts))
    out.append(("bird:heron", fps2,
                [(p[1], p[2], p[3], p[4]) for p in t.pts]))
    return out


def drone_tracks():
    """The 9 continuous single-object birdrone runs, with centroids."""
    got = {}
    for sub in ("train", "valid"):
        for im in glob.glob(os.path.join(HERE, "rf/birdrone", sub, "images", "*")):
            b = os.path.basename(im)
            stem = b.split("_jpg")[0] if "_jpg" in b else b.split(".rf.")[0]
            if not re.match(r"^\d+$", stem):
                continue
            lb = os.path.join(HERE, "rf/birdrone", sub, "labels",
                              os.path.splitext(b)[0] + ".txt")
            if not os.path.exists(lb):
                continue
            g = [l.split() for l in open(lb) if l.strip()]
            g = [x for x in g if len(x) >= 5]
            if len(g) != 1 or int(g[0][0]) != 1:
                continue
            n = int(stem)
            _c, cx, cy, bw, bh = (float(v) for v in g[0][:5])
            got.setdefault(n, (cx*640, cy*640, bw*640, bh*640))
    ks = sorted(got)
    runs, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        (cur.append(b) if b - a == 1 else (runs.append(cur), cur := [b]))
    runs.append(cur)
    out = []
    for i, r in enumerate(r for r in runs if len(r) >= WIN):
        out.append((f"drone:run{i+1}", 25.0, [got[k] for k in r]))
    return out


def windows(tr):
    name, fps, pts = tr
    a = np.array(pts, float)
    out = []
    for s in range(0, len(a) - WIN + 1, STRIDE):
        out.append((name, fps, a[s:s+WIN]))
    return out


# ------------------------------------------------------------ features
def higuchi_fd(x, kmax=8):
    x = np.asarray(x, float); N = len(x)
    L = []
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            Lk.append((np.abs(np.diff(x[idx])).sum() * (N-1)) /
                      (len(idx)-1) / k)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return 1.0
    k = np.arange(1, len(L)+1)
    return float(-np.polyfit(np.log(k), np.log(np.maximum(L, 1e-12)), 1)[0])


def spec_entropy(x, fps):
    x = np.asarray(x, float); x = x - x.mean()
    p = np.abs(np.fft.rfft(x))**2
    p = p[1:]
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def feats_tf(name, fps, a):
    """Proxy of the Drones-2026 feature stack."""
    cx, cy, w, h = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    area = w * h
    v = np.hypot(np.diff(cx), np.diff(cy)) * fps
    acc = np.diff(v) * fps
    return [spec_entropy(area, fps),                       # TF of area (SVE proxy)
            1.0 - spec_entropy(h, fps),                    # peakedness of h
            float(area.std()/max(area.mean(), 1e-9)),      # CV area
            float(v.std()/max(v.mean(), 1e-9)),            # CV velocity
            float(acc.std()/max(abs(acc).mean(), 1e-9)),   # CV accel
            higuchi_fd(cx) + higuchi_fd(cy)]               # trajectory complexity


def feats_traj(name, fps, a):
    """Proxy of trajectory-characterisation methods (no silhouette use)."""
    cx, cy = a[:, 0], a[:, 1]
    v = np.hypot(np.diff(cx), np.diff(cy)) * fps
    path = np.hypot(np.diff(cx), np.diff(cy)).sum()
    net = float(np.hypot(cx[-1]-cx[0], cy[-1]-cy[0]))
    psi = np.arctan2(np.diff(cy), np.diff(cx))
    turn = (np.diff(psi) + np.pi) % (2*np.pi) - np.pi
    return [float(v.std()/max(v.mean(), 1e-9)),
            net/max(path, 1e-9),
            float(np.std(turn)), float(np.mean(np.abs(turn))),
            higuchi_fd(cx) + higuchi_fd(cy)]


def ours(name, fps, a):
    """Fixed physical criterion; returns 'bird' / 'not-bird(unknown)'."""
    for sig in (a[:, 2], a[:, 3]):
        r = FR.band_test(sig, fps)
        if r and r["fired"]:
            return 1
    return 0


# ------------------------------------------------------------ learner
class LogReg:
    def fit(self, X, y, iters=3000, lr=0.1):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1] + 1)
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        for _ in range(iters):
            p = 1/(1+np.exp(-Zb @ w))
            w -= lr * (Zb.T @ (p - y)) / len(y)
        self.w = w

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu) / self.sd
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        return (1/(1+np.exp(-Zb @ self.w)) >= 0.5).astype(int)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    hw = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, max(0, c-hw), min(1, c+hw)


def evaluate(rows, feat_fn, train_idx, test_idx, fixed=None):
    """rows: (name,fps,win). Returns (bird_ok,bird_n,drone_ok,drone_n) on test."""
    y = np.array([1 if r[0].startswith("bird") else 0 for r in rows])
    if fixed is None:
        X = np.array([feat_fn(*r) for r in rows])
        m = LogReg(); m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
    else:
        pred = np.array([fixed(*rows[i]) for i in test_idx])
    yt = y[test_idx]
    b = yt == 1
    return (int((pred[b] == 1).sum()), int(b.sum()),
            int((pred[~b] == 0).sum()), int((~b).sum()))


def main():
    birds = bird_tracks()
    drones = drone_tracks()
    rows, track_of = [], []
    for tr in birds + drones:
        for wnd in windows(tr):
            rows.append(wnd); track_of.append(tr[0])
    track_of = np.array(track_of)
    names = [r[0] for r in rows]
    nb = sum(1 for n in names if n.startswith("bird"))
    print(f"вікна {WIN} кадрів, крок {STRIDE}: птахи {nb} "
          f"(з {len(birds)} треків!), дрони {len(rows)-nb} "
          f"(з {len(drones)} прогонів)")
    print("ЧЕСНО: птахова вибірка крихітна (2 незалежні треки); інтервали Вілсона 95 %.\n")

    METHODS = [("TF-ознаки (Drones'26, проксі)", feats_tf, None),
               ("траєкторний (проксі)", feats_traj, None),
               ("НАШ (фікс. критерій, без навчання)", None, ours)]

    # -------- P1: leave-one-track-out ------------------------------
    print("=" * 86)
    print("П1. LEAVE-ONE-TRACK-OUT (оптимістичний протокол)")
    print("=" * 86)
    print(f"{'метод':>34s} | {'птах: влучень':>14s} | {'дрон: влучень':>14s}")
    print("-" * 74)
    utracks = sorted(set(track_of))
    for label, ff, fx in METHODS:
        bo = bn = do = dn = 0
        for t in utracks:
            te = np.where(track_of == t)[0]
            trn = np.where(track_of != t)[0]
            r = evaluate(rows, ff, trn, te, fixed=fx)
            bo += r[0]; bn += r[1]; do += r[2]; dn += r[3]
        pb = wilson(bo, bn); pd = wilson(do, dn)
        print(f"{label:>34s} | {bo:3d}/{bn:<3d} [{100*pb[1]:.0f};{100*pb[2]:.0f}]% "
              f"| {do:3d}/{dn:<3d} [{100*pd[1]:.0f};{100*pd[2]:.0f}]%")

    # -------- P2: cross-domain --------------------------------------
    print()
    print("=" * 86)
    print("П2. КРОС-ДОМЕН: навчання = кормора́н + прогони дронів 1–5;")
    print("    тест = ЧАПЛЯ (iнше вiдео, вид, фон, 30 Гц) + прогони 6–9")
    print("=" * 86)
    tr_tracks = {"bird:cormorant"} | {f"drone:run{i}" for i in (1, 2, 3, 4, 5)}
    trn = np.where(np.isin(track_of, sorted(tr_tracks)))[0]
    te = np.where(~np.isin(track_of, sorted(tr_tracks)))[0]
    print(f"{'метод':>34s} | {'птах: влучень':>14s} | {'дрон: влучень':>14s}")
    print("-" * 74)
    for label, ff, fx in METHODS:
        bo, bn, do, dn = evaluate(rows, ff, trn, te, fixed=fx)
        pb = wilson(bo, bn); pd = wilson(do, dn)
        print(f"{label:>34s} | {bo:3d}/{bn:<3d} [{100*pb[1]:.0f};{100*pb[2]:.0f}]% "
              f"| {do:3d}/{dn:<3d} [{100*pd[1]:.0f};{100*pd[2]:.0f}]%")

    # -------- P3: box-noise stress (train clean, test noisy) --------
    print()
    print("=" * 86)
    print("П3. СТРЕС ШУМОМ РАМОК: навчання на чистих вікнах П2, тест — ті самі")
    print("    тестові вікна П2 з гаусовим шумом sigma (пкс) на cx, cy, w, h;")
    print(f"    {len(P3_SEEDS)} реалізацій шуму, лічба у вікнах x реалізаціях")
    print("=" * 86)
    y = np.array([1 if r[0].startswith("bird") else 0 for r in rows])
    models = {}
    for label, ff, fx in METHODS:
        if fx is None:
            X = np.array([ff(*rows[i]) for i in trn])
            m = LogReg(); m.fit(X, y[trn]); models[label] = m
    print(f"{'sigma':>6s} | {'метод':>34s} | {'птах→«птах»':>12s} | {'птах→unknown':>12s} "
          f"| {'дрон→«птах»':>12s}")
    print("-" * 90)
    for sigma in P3_SIGMAS:
        for label, ff, fx in METHODS:
            bb = bu = db = nb_ = nd_ = 0
            for seed in P3_SEEDS:
                rng = np.random.default_rng(seed)
                for i in te:
                    name, fps, a = rows[i]
                    an = a + rng.normal(0.0, sigma, a.shape) if sigma > 0 else a.copy()
                    an[:, 2:4] = np.maximum(an[:, 2:4], 1.0)
                    if fx is None:
                        p = int(models[label].predict([ff(name, fps, an)])[0])
                    else:
                        p = fx(name, fps, an)
                    if name.startswith("bird"):
                        nb_ += 1; bb += p; bu += (1 - p) if fx is not None else 0
                    else:
                        nd_ += 1; db += p
            unk = f"{bu:3d}/{nb_:<3d}" if fx is not None else "  немає опції"
            print(f"{sigma:6.1f} | {label:>34s} | {bb:4d}/{nb_:<5d} | {unk:>12s} | {db:4d}/{nd_:<5d}")
        print("-" * 90)

    # -------- P4: decoy tracks (truth: no object) ------------------
    dpath = os.path.join(HERE, "reports/decoy_tracks.pkl")
    print()
    print("=" * 86)
    print("П4. ТРЕКИ-ПУСТИШКИ: хибні треки детектора на воді (alp_on_failures.py),")
    print("    істина — жодного об'єкта; ті самі моделі, що в П3 (навчання — чисті вікна П2)")
    print("=" * 86)
    if not os.path.exists(dpath):
        print("  немає reports/decoy_tracks.pkl -- спершу запустіть alp_on_failures.py")
    else:
        import pickle
        decoys = pickle.load(open(dpath, "rb"))
        drows = [w for tr in decoys for w in windows(tr)]
        print(f"  пустишок: {len(decoys)} треків -> {len(drows)} вікон {WIN} кадрів "
              f"(вікна одного треку не незалежні)")
        print(f"{'метод':>34s} | {'«птах»':>7s} | {'«дрон»':>7s} | {'unknown':>8s}")
        print("-" * 70)
        for label, ff, fx in METHODS:
            if fx is None:
                p = models[label].predict([ff(*r) for r in drows])
                nb = int(p.sum()); nd = len(p) - nb
                print(f"{label:>34s} | {nb:3d}/{len(drows):<3d} | {nd:3d}/{len(drows):<3d} | {'немає опції':>8s}")
            else:
                nb = sum(fx(*r) for r in drows)
                print(f"{label:>34s} | {nb:3d}/{len(drows):<3d} | {0:3d}/{len(drows):<3d} | "
                      f"{len(drows) - nb:3d}/{len(drows):<3d}")

    print()
    print("застереження: проксі-реалізації наші, не авторський код; висновок —")
    print("про архітектури (навчені ознаки проти фіксованого фізичного критерію).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""END-TO-END: real detector -> symbol grounding -> features -> ALP explanation.

Chain (the practical realisation of the method):

  frames --YOLO--> bbox  --deprojection--> bearing + range --> metric track
         --causal tracker--> kinematic facts --> air_objects.pl --> class + PROOF

The deprojection is the same pinhole relation used by the companion study:
    beta = arctan((u - cx) / f_px)                 bearing to the object
    R    = k / w_px                                range from apparent width
so the grounding error sigma of that study is the error of THIS step.

Output per window: what the detector claims, what the kinematics admit, and the
logical justification for confirming or revising the detector's label.
"""
from __future__ import annotations

import argparse, glob, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat


# --------------------------------------------------------------------------
# 1. detection
# --------------------------------------------------------------------------
def detect_track(frames, weights, conf, imgsz, gate_px=60.0, coast=6):
    """Detector + GATED DATA ASSOCIATION -> one coherent track.

    Raw detector output is NOT a track: on a 12-px object the highest-confidence
    box in successive frames is often a different object or a false positive,
    producing 200-500 px jumps. Kinematics computed on that are meaningless, so
    association is mandatory before any reasoning. Nearest-neighbour gating:
    accept the candidate closest to the constant-velocity prediction, and only
    if it falls inside `gate_px`; otherwise coast on the prediction.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    out, last, vel, miss = [], None, np.zeros(2), 0
    for f in frames:
        r = model.predict(source=f, verbose=False, conf=conf, imgsz=imgsz)[0]
        cands = []
        if r.boxes is not None and len(r.boxes):
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].tolist()
                cands.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                              max(1.0, x2 - x1), float(r.boxes.conf[j])))
        if last is None:                       # track initiation: most confident
            pick = max(cands, key=lambda c: c[3]) if cands else None
        else:
            pred = np.array(last[:2]) + vel
            pick, best = None, gate_px
            for c in cands:                    # nearest to prediction, gated
                d = float(np.hypot(c[0] - pred[0], c[1] - pred[1]))
                if d < best:
                    best, pick = d, c
        if pick is None:
            miss += 1
            out.append(None)
            if miss > coast:                   # track lost -> allow re-init
                last, vel, miss = None, np.zeros(2), 0
            continue
        if last is not None:
            vel = 0.5 * vel + 0.5 * (np.array(pick[:2]) - np.array(last[:2]))
        last, miss = pick, 0
        out.append(pick)
    return out


# --------------------------------------------------------------------------
# 2. symbol grounding: pixels -> metric position (camera frame)
# --------------------------------------------------------------------------
def ground(dets, img_w, img_h, hfov_deg, range_k):
    """Deproject each detection into a camera-relative metric point."""
    f_px = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx0, cy0 = img_w / 2.0, img_h / 2.0
    pts, mask = [], []
    for d in dets:
        if d is None:
            pts.append([np.nan] * 3); mask.append(False); continue
        u, v, w_px, _ = d
        beta = math.atan((u - cx0) / f_px)       # azimuth   (formula (1))
        eps = math.atan((v - cy0) / f_px)        # elevation (formula (1))
        R = range_k / w_px                       # range from apparent size
        # camera frame: x forward, y right, z up
        pts.append([R * math.cos(beta) * math.cos(eps),
                    R * math.sin(beta) * math.cos(eps),
                    -R * math.sin(eps)])
        mask.append(True)
    return np.array(pts), np.array(mask)


def interpolate_gaps(pos, mask, max_gap=5):
    """Bridge short detection gaps (a tracker would coast through them)."""
    p = pos.copy()
    idx = np.arange(len(p))
    for c in range(3):
        good = mask & np.isfinite(p[:, c])
        if good.sum() < 2:
            continue
        p[:, c] = np.interp(idx, idx[good], p[good, c])
    return p


# --------------------------------------------------------------------------
# 3. windowed hover: net displacement, not instantaneous speed
#    (a DC quantity averages noise down as sigma/sqrt(N); instantaneous speed
#     multiplies it by 1/dt -- see the study's estimator analysis)
# --------------------------------------------------------------------------
def hover_fraction(pos, dt, win_s=2.0, drift_max=1.0):
    w = max(4, int(win_s / dt))
    n = len(pos) // w
    if n < 2:
        return 0.0
    means = np.array([pos[i * w:(i + 1) * w].mean(axis=0) for i in range(n)])
    drift = np.linalg.norm(np.diff(means[:, :2], axis=0), axis=1) / win_s
    return float((drift < drift_max).mean())


# --------------------------------------------------------------------------
# 4. ALP
# --------------------------------------------------------------------------
FACTS = ("f_det(_,_)", "f_speed(_)", "f_speed_cv(_)", "f_hover(_)",
         "f_flap(_)", "f_turn_irreg(_)", "f_straight(_)", "f_turn_bias(_)")


class Abducer:
    def __init__(self, pl_path):
        from pyswip import Prolog
        self.p = Prolog()
        self.p.consult(pl_path)

    def set_thr(self, name, value):
        """Override one authored threshold (for the sensitivity analysis)."""
        list(self.p.query(f"retractall(thr({name},_))"))
        self.p.assertz(f"thr({name},{float(value):.5f})")

    def explain(self, f, det_class, det_conf):
        for x in FACTS:
            list(self.p.query(f"retractall({x})"))
        if det_class:
            self.p.assertz(f"f_det({det_class},{det_conf:.3f})")
        # A fact is asserted ONLY when it is actually groundable. An unstated
        # fact activates no constraint, which is the formalism's native way to
        # represent an unobservable quantity (rather than asserting noise).
        for name in ("f_speed", "f_speed_cv", "f_hover",
                     "f_turn_irreg", "f_straight", "f_turn_bias"):
            v = f.get(name)
            if v is not None:
                self.p.assertz(f"{name}({float(v):.4f})")
        if f.get("f_flap") is not None:
            self.p.assertz(f"f_flap({'true' if f['f_flap'] else 'false'})")
        r = list(self.p.query("classify(C, W)"))
        return (str(r[0]["C"]), str(r[0]["W"])) if r else ("unknown", "no_solution")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--weights", default="models/detector.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--range-k", type=float, default=2016.0)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--window", type=int, default=100, help="frames per decision")
    args = ap.parse_args()

    fs = sorted(f for f in glob.glob(os.path.join(args.frames, "*.jpg"))
                if re.match(r"^\d+\.jpg$", os.path.basename(f)))[:args.n]
    if not fs:
        sys.exit("no numbered frames found")
    from PIL import Image
    W, H = Image.open(fs[0]).size
    dt = 1.0 / args.fps

    print(f"кадрів: {len(fs)}  ({W}x{H})   детектор: {os.path.basename(args.weights)}")
    dets = detect_track(fs, args.weights, args.conf, args.imgsz)
    n_hit = sum(d is not None for d in dets)
    print(f"детекцій: {n_hit}/{len(dets)} ({100*n_hit/len(dets):.0f}%)")
    if n_hit < 20:
        sys.exit("too few detections")

    pos, mask = ground(dets, W, H, args.hfov, args.range_k)
    pos = interpolate_gaps(pos, mask)
    rng_m = np.linalg.norm(pos[:, :2], axis=1)
    print(f"дальність за депроекцією: {np.nanmin(rng_m):.0f}..{np.nanmax(rng_m):.0f} м "
          f"(медіана {np.nanmedian(rng_m):.0f} м)")

    abd = Abducer(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "air_objects.pl"))
    print(f"\n{'вікно':>12s} {'дет.':>6s} {'v,м/с':>7s} {'hover':>6s} {'flap':>5s} "
          f"{'irreg':>6s}  РІШЕННЯ / ОБҐРУНТУВАННЯ")
    print("-" * 108)
    w = args.window
    for s in range(0, len(pos) - w + 1, w):
        seg = pos[s:s + w]
        f = feat.extract(seg, dt, 0.0, np.random.default_rng(0))
        f["f_hover"] = hover_fraction(seg, dt)          # robust DC estimator
        confs = [d[3] for d in dets[s:s + w] if d is not None]
        dc = float(np.mean(confs)) if confs else 0.0
        cls, why = abd.explain(f, "drone" if confs else None, dc)
        ti = "—".rjust(6) if f["f_turn_irreg"] is None else f"{f['f_turn_irreg']:6.2f}"
        print(f"{s:5d}-{s+w:5d} {dc:6.2f} {f['f_speed']:7.2f} {f['f_hover']:6.2f} "
              f"{str(f['f_flap']):>5s} {ti}  {cls:8s} {why}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# 5. ANGULAR-DOMAIN features (the monocular-honest path)
# --------------------------------------------------------------------------
# Measured on real detections at ~170 m: bearing noise is small, but range from
# apparent width carries ~27 % error on a 12-px box, i.e. tens of metres. Metric
# speed derived from it is meaningless (200-700 m/s for a slow quadrotor).
# Therefore kinematics are computed in the ANGULAR domain, where the measurement
# is good, and `f_speed` is simply NOT ASSERTED. In abduction an unstated fact
# activates no constraint, so speed-envelope refutations correctly stay silent
# instead of firing on noise -- the formalism's native way to handle an
# unobservable quantity.
def angular_features(dets, img_w, img_h, hfov_deg, dt,
                     hold_win_s=2.0, hold_rate=0.010, hold_size=0.10):
    f_px = (img_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    ok = [(i, d) for i, d in enumerate(dets) if d is not None]
    if len(ok) < 20:
        return None
    idx = np.array([i for i, _ in ok], float)
    bx = np.array([math.atan((d[0] - img_w / 2.0) / f_px) for _, d in ok])
    by = np.array([math.atan((d[1] - img_h / 2.0) / f_px) for _, d in ok])
    wp = np.array([d[2] for _, d in ok], float)

    tt = idx * dt
    dts = np.diff(tt)
    dts[dts <= 0] = dt
    rate_x = np.diff(bx) / dts
    rate_y = np.diff(by) / dts
    ang_rate = np.hypot(rate_x, rate_y)                  # rad/s across LOS
    psi = np.arctan2(rate_y, rate_x)
    turn = _wrap_arr(np.diff(psi)) / dts[1:] if len(psi) > 1 else np.zeros(1)

    # "holds position" in the angular domain: bearing barely moves AND apparent
    # size barely changes -> the object is neither crossing nor closing.
    w_win = max(4, int(hold_win_s / dt))
    n_win = len(bx) // w_win
    held = []
    for i in range(n_win):
        sl = slice(i * w_win, (i + 1) * w_win)
        d_ang = float(np.hypot(bx[sl][-1] - bx[sl][0], by[sl][-1] - by[sl][0])) / hold_win_s
        d_sz = abs(float(wp[sl][-1] - wp[sl][0])) / max(1.0, float(wp[sl].mean()))
        held.append(d_ang < hold_rate and d_sz < hold_size)
    # GROUNDABILITY GATES -- the same ones `features.py::extract` applies. They
    # were missing here, and the omission was not cosmetic: on real footage the
    # tracks that survive association are wave crests, whose apparent motion is
    # erratic, so an ungated `f_turn_irreg` satisfies `irregular_manoeuvring`
    # and the theory confidently calls a wave a bird. A signature must entail
    # OBJECT BEHAVIOUR, not measurement chaos -- the same failure mode as the
    # absence-signature failure (see the article), arriving by another route.
    #
    # The test is the frequency principle used throughout: manoeuvring is
    # band-limited, grounding noise is broadband and flat, so power density
    # above the manoeuvre band estimates the noise floor across the whole band.
    # A quantity whose variance is mostly noise is not a fact.
    import features as _feat
    fps = 1.0 / dt
    turn_var, turn_noise = _feat._denoise_var(turn, dt) if len(turn) >= 32 \
        else (float(np.var(turn)), 1.0)
    rate_var, rate_noise = _feat._denoise_var(ang_rate, dt) if len(ang_rate) >= 32 \
        else (float(np.var(ang_rate)), 1.0)
    turn_ok = turn_noise <= _feat.NOISE_SHARE_MAX
    rate_ok = rate_noise <= _feat.NOISE_SHARE_MAX

    return {
        # hover is a DC quantity (net angular displacement) -- always groundable
        "f_hover": float(np.mean(held)) if held else 0.0,
        "f_turn_irreg": float(np.sqrt(turn_var)) if turn_ok else None,
        "f_turn_bias": float(np.mean(turn)) if turn_ok else None,
        "f_straight": float((np.abs(turn) < 0.05).mean()) if turn_ok else None,
        # `f_speed_cv` is a fact about AIRSPEED fluctuation. Angular rate is not
        # airspeed: it mixes speed with range, so an object at varying range
        # shows a varying angular rate at constant airspeed. Range is exactly
        # the quantity this pipeline refuses to trust (27 % error), which is why
        # `f_speed` is already withheld here -- and a fact derived from the same
        # unavailable quantity must be withheld for the same reason. Asserting it
        # let two wave tracks satisfy `fluctuating_airspeed` and be called birds.
        # The name of a fact must promise no more than the measurement delivers.
        "f_speed_cv": None,
        "f_flap": None,           # supplied by the silhouette channel, not here
        "_ang_rate": float(ang_rate.mean()),
        "_turn_noise": turn_noise,
        "_n": len(ok),
    }


def _wrap_arr(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

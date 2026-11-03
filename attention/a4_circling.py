#!/usr/bin/env python3
"""E15. Heading facts for the ALP of Article 3 (soaring_circling and the other heading facts),
measured in the image plane (protocol fixed before the run).

Heading facts follow features.extract of Article 3 exactly (EMA 0.5 position smoother, backward
difference, velocity EMA 0.3, heading rate, noise floor above 6 Hz removed, groundability: the
displacement over the velocity window must exceed 3x the displacement noise, and the noise share of
the heading-rate variance must not exceed 0.5). Leg (b) of the Article 3 heading guard (speed above
1 m/s) needs metres and is not groundable in pixels; it is not applied. Heading rate in the image
plane does not depend on range, so the facts are groundable for a FIXED camera: the Midway camera
is fixed; the DUT camera may follow the vehicle, so for UAVs the facts are NOT asserted in the main
variant and asserted only in a worst-case variant.
Facts: f_turn_irreg, f_straight, f_turn_bias (as in Article 3). Signatures that use them in
air_objects.pl: bird/soaring_circling, bird/irregular_manoeuvring (needs f_hover, not asserted),
drone/sustained_straight_run.
Part 1: reference trajectories. Part 2: the system of E11 (window + abductive identity + classifier
-> ALP), facts measured on the held positions of the longest held run.
Output: reports/e15_circling.txt, reports/e15_circling.json
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPO, REPORTS, WIN, frames, load   # noqa: E402
import a4_drones as D                                                # noqa: E402
import a4_experiments as X                                           # noqa: E402
import a4_v6_runs as V                                               # noqa: E402
sys.path.insert(0, REPO)
from features import EMA, V_EMA, HEADING_SNR, NOISE_SHARE_MAX, STRAIGHT_RATE, _denoise_var, _wrap   # noqa: E402


def heading_facts(pos, dt):
    """features.extract heading path of Article 3 on image-plane positions (px)."""
    obs = np.asarray(pos, float)
    if len(obs) < WIN:
        return None
    sm = np.empty_like(obs); sm[0] = obs[0]
    for k in range(1, len(obs)):
        sm[k] = EMA * obs[k] + (1 - EMA) * sm[k - 1]
    v = np.zeros_like(sm); v[1:] = (sm[1:] - sm[:-1]) / dt; v[0] = v[1]
    vs = np.empty_like(v); vs[0] = v[0]
    for k in range(1, len(v)):
        vs[k] = V_EMA * v[k] + (1 - V_EMA) * vs[k - 1]
    jitter = float(np.std(np.diff(obs, n=2, axis=0)) / np.sqrt(6.0))
    psi = np.arctan2(vs[:, 1], vs[:, 0]); rate = _wrap(np.diff(psi)) / dt
    travel = float(np.median(np.linalg.norm(vs, axis=1))) * dt * (2.0 - V_EMA) / V_EMA
    heading_ok = travel > HEADING_SNR * jitter * np.sqrt(2.0)
    rate_var, share = _denoise_var(rate, dt)
    if not (heading_ok and share <= NOISE_SHARE_MAX):
        return None
    return dict(f_turn_irreg=float(np.sqrt(rate_var)), f_straight=float((np.abs(rate) < STRAIGHT_RATE).mean()),
                f_turn_bias=float(rate.mean()))


class Alp3H(V.Alp3):
    def classify_h(self, det, flap, hf):
        for f in ("f_det(_,_)", "f_flap(_)", "f_speed(_)", "f_speed_cv(_)", "f_hover(_)", "f_turn_irreg(_)",
                  "f_straight(_)", "f_turn_bias(_)"):
            list(self.pl.query(f"retractall({f})"))
        if det is not None:
            self.pl.assertz(f"f_det({det[0]},{det[1]})")
        if flap is not None:
            self.pl.assertz(f"f_flap({'true' if flap else 'false'})")
        for k, v in (hf or {}).items():
            self.pl.assertz(f"{k}({v})")
        r = list(self.pl.query("classify(C, J)", maxresult=1))[0]
        sig = [str(x["W"]) for x in self.pl.query("admissible(bird, W)")]
        return str(r["C"]), str(r["J"]).split("(")[0], sig


def circling(h):
    return h is not None and abs(h["f_turn_bias"]) > 0.10 and h["f_turn_irreg"] <= 0.35


def straight(h):
    return h is not None and h["f_straight"] > 0.70


def run_positions():
    """Re-run the E11 system holder and keep its positions (series must equal v6_runs)."""
    R6 = load(os.path.join(CACHE, "v6_runs.pkl")); out = {}
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl")); det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    hs = {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            h = hs.setdefault(tid, X.Holder(truth[tid][k], "self", alp=True))
            h.step(fr, None, None, det.get(k, []))
    for tid, h in hs.items():
        out[("bird", tid)] = (list(h.series), list(h.pos))
    detu = load(os.path.join(CACHE, "e6_det_uav_ours.pkl"))
    for i, idx in D.segments():
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            im0 = cv2.imread(fs[idx[0]]); W, H = int(round(im0.shape[1] / k)), int(round(im0.shape[0] / k))
            h = X.Holder(((gt[idx[0]][0] + gt[idx[0]][2] / 2) / k, (gt[idx[0]][1] + gt[idx[0]][3] / 2) / k), "self", alp=True)
            for j in idx:
                h.step(cv2.resize(cv2.imread(fs[j]), (W, H), interpolation=cv2.INTER_AREA), None, None, detu[(i, idx[0], S, j)])
            out[("uav", (i, idx[0], S))] = (list(h.series), list(h.pos))
    same = all(np.array_equal(np.array(out[k][0], float), np.array(R6[k]["sys"]["series"], float), equal_nan=True) for k in out)
    return out, same


def held_run_positions(series, pos):
    a = np.array(series, float); ok = ~np.isnan(a[:, 0]); best = cur = 0; end = -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    return np.array(pos[end - best + 1:end + 1], float) if best >= WIN else None


def main():
    L, J = [], {}
    # ---------------- part 1: reference trajectories
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl")); T = load(os.path.join(CACHE, "truth_series_birds.pkl"))
    import a4_soar as SO
    b = []
    for t in tracks:
        hf = heading_facts([(p[1], p[2]) for p in t["pts"]], 1 / X.FPS_MIDWAY)
        r = SO.longest(T[t["id"]]); fl = bool(r is not None and SO.fired(r, SO.FPS_B))
        b.append((hf, fl))
    u = []
    for i, idx in D.segments():
        fs, gt = D.load_seq(i)
        for S in D.SIZES:
            k = float(np.median([max(gt[j][2], gt[j][3]) for j in idx])) / S
            u.append(heading_facts([((gt[j][0] + gt[j][2] / 2) / k, (gt[j][1] + gt[j][3] / 2) / k) for j in idx], 1 / D.FPS))
    soar = [h for h, f in b if not f]
    L.append("Part 1, reference trajectories (Midway camera fixed; DUT camera possibly following)")
    L.append(f"  birds {len(b)}: heading groundable {sum(h is not None for h, _ in b)}; soaring (no flap) {len(soar)}, of them "
             f"groundable {sum(h is not None for h in soar)}, circling signature {sum(circling(h) for h in soar)}, "
             f"straight-run (drone signature) {sum(straight(h) for h in soar)}")
    L.append(f"  UAV runs {len(u)} (worst case, as if the camera were fixed): groundable {sum(h is not None for h in u)}, "
             f"circling {sum(circling(h) for h in u)}, straight-run {sum(straight(h) for h in u)}")
    gb = [h for h, _ in b if h is not None]
    if gb:
        L.append(f"  bird |turn bias| median {np.median([abs(h['f_turn_bias']) for h in gb]):.3f} rad/s, turn irreg median "
                 f"{np.median([h['f_turn_irreg'] for h in gb]):.3f}, straight median {np.median([h['f_straight'] for h in gb]):.2f}")
    J["p1"] = dict(birds=len(b), birds_ground=sum(h is not None for h, _ in b), soar=len(soar),
                   soar_circ=sum(circling(h) for h in soar), soar_straight=sum(straight(h) for h in soar),
                   uav=len(u), uav_ground=sum(h is not None for h in u), uav_circ=sum(circling(h) for h in u),
                   uav_straight=sum(straight(h) for h in u))
    # ---------------- part 2: in the system
    P, same = run_positions()
    L.append(""); L.append(f"Part 2, system of E11 (positions re-run, identical to v6_runs: {same})")
    R6 = load(os.path.join(CACHE, "v6_runs.pkl")); alp = Alp3H()
    for c in ("bird", "uav"):
        for variant in (("no heading facts", False), ("heading facts", True)):
            if c == "bird" and variant[1] is False:
                pass
            labs, kinds, circ, strt, grd = [], [], 0, 0, 0
            for k in [k for k in R6 if k[0] == c]:
                rec = R6[k]["sys"]; votes = rec["votes"]
                if votes and max(cf for _, cf in votes) >= V.CONF_SURE:
                    labs.append(max(votes, key=lambda v: v[1])[0]); kinds.append("classifier"); continue
                det = None
                if len(votes) >= 5:
                    cls = max(("bird", "drone"), key=lambda q: sum(v == q for v, _ in votes))
                    det = (cls, float(np.median([cf for v, cf in votes if v == cls])))
                run = V.longest_run(rec["series"]); fps = X.FPS_MIDWAY if c == "bird" else D.FPS
                flap = None if run is None else any((r := V.band_test(run[:, q], fps)) and r["fired"] for q in (0, 1))
                hf = None
                if variant[1]:
                    pos = held_run_positions(*P[k])
                    hf = heading_facts(pos, 1 / fps) if pos is not None else None
                    grd += hf is not None; circ += circling(hf); strt += straight(hf)
                lab, how, _ = alp.classify_h(det, flap, hf)
                labs.append(lab); kinds.append(how)
            truth = "bird" if c == "bird" else "drone"
            d = dict(n=len(labs), correct=labs.count(truth), unknown=labs.count("unknown"),
                     wrong=len(labs) - labs.count(truth) - labs.count("unknown"),
                     d2b=labs.count("bird") if c == "uav" else 0, groundable=grd, circling=circ, straight=strt,
                     kinds={q: kinds.count(q) for q in sorted(set(kinds))})
            J[f"p2_{c}_{'h' if variant[1] else 'noh'}"] = d
            tag = variant[0] + (" (worst case for UAV)" if c == "uav" and variant[1] else "")
            L.append(f"  {c:4s} {tag:38s} correct {d['correct']:2d} wrong {d['wrong']:2d} unknown {d['unknown']:2d}"
                     + (f" drone->bird {d['d2b']}" if c == "uav" else "")
                     + (f" | groundable {grd}, circling {circ}, straight {strt}" if variant[1] else "") + f"  {d['kinds']}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e15_circling.txt"), "w").write(t + "\n")
    json.dump(J, open(os.path.join(REPORTS, "e15_circling.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

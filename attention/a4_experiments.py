#!/usr/bin/env python3
"""Article 4 experiments on the evaluation set (a4_select.py) with the cached
detector boxes (a4_detcache.py). One sequential pass over the Midway clip.

E1  classifier confidence as a time series: for every frame of every track,
    the maximum detector confidence (any class) inside a 45 px gate around the
    true position. Coverage and longest hole at thresholds 0.5 / 0.25 / 0.1,
    and whether tracking-by-detection (coast 5) keeps one continuous
    fragment of >= 48 frames.
E2  visibility in the attention window: is there a negative-contrast blob
    within 25 px of the true position when the threshold is taken from a
    120x120 window around it (all frames / frames inside detector holes).
E3  closed-loop holding (truth is used only to assign the target once and to
    score): six holding strategies, 48-frame spectral decision on the held
    series, decision scored as correct if >= 50 % of its window is on target.

Outputs: reports/e1.csv, reports/e2.csv, reports/e3.csv, reports/summary.txt
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MIDWAY, CACHE, REPORTS, FPS_MIDWAY, WIN, band_test, frames,   # noqa: E402
                    local_blobs, load)

GATE_DET, ON_TARGET, COAST = 45.0, 25.0, 5
THRS = (0.5, 0.25, 0.1)
FLASH = 0.25
ID_HIST, ID_MIN, ID_SIZE, ID_PEAK = 15, 5, (0.5, 2.0), (0.4, 2.5)   # E9 identity check (fixed a priori)
ALP_FLOOR = 3.0   # E10: body-length floor of the kinematic constraint, px (a4_track_alp.pl)
SEEDS = (7, 11, 23, 42, 77)
SIZE_BINS = ((4, 6), (7, 9), (10, 12))


# ----------------------------------------------------------------- helpers
def holes(vis):
    """longest run of False."""
    best = cur = 0
    for v in vis:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def tbd_longest(vis, coast=COAST):
    """tracking-by-detection: a fragment survives gaps <= coast; returns the
    longest fragment measured in consecutive DETECTED frames (the series a
    spectral test can use has no gaps)."""
    best = cur = 0
    for v in vis:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def decide(series, on):
    """first 48-frame window without gaps where band_test fires on w or h.
    Returns (fired, correct, start_index)."""
    a = np.array(series, float)
    for s in range(0, len(a) - WIN + 1):
        w = a[s:s + WIN]
        if np.isnan(w).any():
            continue
        for col in (0, 1):
            r = band_test(w[:, col], FPS_MIDWAY)
            if r and r["fired"]:
                return True, bool(np.mean(on[s:s + WIN]) >= 0.5), s
    return False, False, -1


class Holder:
    """attention ladder: prediction -> coast (<= 5 frames) -> cue only after a
    confirmed loss. cue_mode:
      none        -- no cue after the initial assignment;
      self        -- the classifier's own flash (conf >= FLASH) NEAREST TO THE
                     HOLDER'S OWN PREDICTION (or to the extrapolated last position
                     after a loss, with a gate widening 2 px per frame, max 150);
                     no ground truth is used;
      self_oracle -- the flash attributed to the target by the TRUE position
                     (upper bound, reported only for comparison);
      ext         -- external cue: true position + Gaussian noise, every `period` frames.
    selector: nearest | contrast."""

    def __init__(self, start, cue_mode, selector="nearest", period=0, sigma=0.0, seed=0, identity=False, alp=False):
        self.last = np.array(start, float)
        self.vel = np.zeros(2)
        self.miss = 0
        self.cue = None
        self.cue_mode, self.selector = cue_mode, selector
        self.period, self.sigma = period, sigma
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.lost_xy, self.since = None, 0
        self.series, self.pos = [], []
        self.trace = []    # per frame: prediction, window centre, state (for figures only)
        self.identity = identity   # E9: accept only silhouettes consistent with the target model
        self.hist = []             # (size, contrast) of the last ID_HIST accepted silhouettes
        self.alp = alp             # E10: abductive association (a4_track_alp.pl), kinematic constraint
        self.sizes = []            # sizes of the last ID_HIST accepted silhouettes (body length)
        self.alp_log = []          # per frame: (prediction, candidates, chosen id or None) for the Prolog cross-check

    def _alp_choose(self, bl, pred):
        """Python twin of explain/2 in a4_track_alp.pl: nearest candidate whose residual does not
        exceed one body length (median size of the last accepted silhouettes, floor 3 px); with
        fewer than ID_MIN accepted silhouettes the 30 px window gate applies; None = 'unseen'."""
        lim = max(float(np.median(self.sizes)), ALP_FLOOR) if len(self.sizes) >= ID_MIN else 30.0
        ok = [b for b in bl if np.hypot(b[0] - pred[0], b[1] - pred[1]) <= lim]
        pick = min(ok, key=lambda b: np.hypot(b[0] - pred[0], b[1] - pred[1])) if ok else None
        self.alp_log.append((tuple(pred), [(b[0], b[1], max(b[2], b[3])) for b in bl],
                             float(np.median(self.sizes)) if len(self.sizes) >= ID_MIN else None,
                             None if pick is None else bl.index(pick)))
        return pick

    def _consistent(self, b):
        """E9 identity check: size max(w,h) within ID_SIZE and contrast within ID_PEAK of the
        medians of the last ID_HIST accepted silhouettes (enforced from ID_MIN samples on)."""
        if not self.identity or len(self.hist) < ID_MIN:
            return True
        s0 = np.median([h[0] for h in self.hist]); p0 = np.median([h[1] for h in self.hist])
        s, p = max(b[2], b[3]), b[4]
        return ID_SIZE[0] * s0 <= s <= ID_SIZE[1] * s0 and ID_PEAK[0] * p0 <= p <= ID_PEAK[1] * p0

    def step(self, fr, truth_xy, flash_xy, dets=()):
        if self.cue_mode == "ext" and self.n % self.period == 0:
            self.cue = np.array(truth_xy) + self.rng.normal(0, self.sigma, 2)
        elif self.cue_mode == "self_oracle" and flash_xy is not None:
            self.cue = np.array(flash_xy)
        elif self.cue_mode == "self":
            if self.last is not None:
                anchor, gate = self.last + self.vel, GATE_DET
            elif self.lost_xy is not None:
                anchor = self.lost_xy + self.vel * self.since
                gate = min(GATE_DET + 2.0 * self.since, 150.0)
            else:
                anchor = None
            if anchor is not None:
                c = [d for d in dets if d[4] >= FLASH and np.hypot(d[0] - anchor[0], d[1] - anchor[1]) < gate]
                if c:
                    d = min(c, key=lambda d: np.hypot(d[0] - anchor[0], d[1] - anchor[1]))
                    self.cue = np.array((d[0], d[1]))
        self.n += 1
        pick, pred, win, state = None, None, None, None
        if self.last is not None:
            pred = self.last + self.vel
            win, state = pred, "hold"
            bl = [b for b in local_blobs(fr, pred[0], pred[1])
                  if np.hypot(b[0] - pred[0], b[1] - pred[1]) < 30.0 and self._consistent(b)]
            if self.alp:
                pick = self._alp_choose(bl, pred) if bl else None
            elif bl:
                pick = (max(bl, key=lambda b: b[4]) if self.selector == "contrast"
                        else min(bl, key=lambda b: np.hypot(b[0] - pred[0], b[1] - pred[1])))
        if pick is None and self.cue is not None and (self.last is None or self.miss >= COAST):
            win = self.cue.copy()
            bl = [b for b in local_blobs(fr, self.cue[0], self.cue[1])
                  if np.hypot(b[0] - self.cue[0], b[1] - self.cue[1]) < 45.0 and self._consistent(b)]
            if bl:
                pick = (max(bl, key=lambda b: b[4]) if self.selector == "contrast"
                        else min(bl, key=lambda b: np.hypot(b[0] - self.cue[0], b[1] - self.cue[1])))
                self.vel = np.zeros(2); state = "reacq"
        if pick is not None:
            if self.last is not None:
                self.vel = 0.6 * self.vel + 0.4 * (np.array(pick[:2]) - self.last)
            self.last = np.array(pick[:2]); self.miss = 0
            self.lost_xy, self.since = None, 0
            self.series.append((pick[2], pick[3])); self.pos.append(tuple(pick[:2]))
            if self.identity:
                self.hist = (self.hist + [(max(pick[2], pick[3]), pick[4])])[-ID_HIST:]
            if self.alp:
                self.sizes = (self.sizes + [max(pick[2], pick[3])])[-ID_HIST:]
        else:
            self.miss += 1
            if self.last is not None and self.miss <= COAST:
                self.last = self.last + self.vel; state = "coast"
            else:
                state = "lost"
                if self.last is not None:
                    self.lost_xy, self.since = self.last.copy(), 0
                self.last = None
                self.since += 1
            self.series.append((np.nan, np.nan)); self.pos.append(None)
        self.trace.append(dict(pred=pred, win=win, pick=pick, state=state))


class TBD:
    """tracking-by-detection baseline on detector boxes (conf >= FLASH)."""

    def __init__(self, start):
        self.last = np.array(start, float); self.vel = np.zeros(2); self.miss = 0
        self.series, self.pos = [], []

    def step(self, dets):
        pick = None
        if self.last is not None:
            pred = self.last + self.vel
            c = [d for d in dets if d[4] >= FLASH and np.hypot(d[0] - pred[0], d[1] - pred[1]) < GATE_DET]
            if c:
                pick = min(c, key=lambda d: np.hypot(d[0] - pred[0], d[1] - pred[1]))
        if pick is not None:
            self.vel = 0.6 * self.vel + 0.4 * (np.array(pick[:2]) - self.last)
            self.last = np.array(pick[:2]); self.miss = 0
            self.series.append((pick[2], pick[3])); self.pos.append(tuple(pick[:2]))
        else:
            self.miss += 1
            if self.last is not None and self.miss <= COAST:
                self.last = self.last + self.vel
            else:
                self.last = None
            self.series.append((np.nan, np.nan)); self.pos.append(None)


def modes_for(start):
    out = [("tbd", None, TBD(start))]
    out.append(("att_none", None, Holder(start, "none")))
    out.append(("att_self", None, Holder(start, "self")))
    out.append(("att_self_oracle", None, Holder(start, "self_oracle")))
    out.append(("att_contrast_self", None, Holder(start, "self", selector="contrast")))
    for sd in SEEDS:
        out.append(("att_ext0.2Hz", sd, Holder(start, "ext", period=int(5 * FPS_MIDWAY), sigma=15.0, seed=sd)))
        out.append(("att_ext1Hz", sd, Holder(start, "ext", period=int(FPS_MIDWAY), sigma=15.0, seed=sd)))
    return out


# -------------------------------------------------------------------- main
def main():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {t["id"]: {int(p[0]): p for p in t["pts"]} for t in tracks}
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in tracks}
    by_frame = {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)

    conf = {tid: [] for tid in truth}          # E1: (k, max conf in gate)
    vis = {tid: [] for tid in truth}           # E2: (k, local blob near truth)
    state = {}                                 # E3: tid -> [(mode, seed, obj)]
    on = {}                                    # E3: (tid, mode, seed) -> [on target]
    fl = {}                                    # E3: (tid, mode, seed) -> [classifier flash in frame]
    f_last = max(by_frame)
    for k, fr in frames(MIDWAY, 0, f_last):
        for tid in by_frame.get(k, []):
            p = truth[tid][k]
            tx, ty = p[1], p[2]
            near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < GATE_DET]
            c = max((d[4] for d in near), default=0.0)
            conf[tid].append((k, c))
            vis[tid].append((k, any(np.hypot(b[0] - tx, b[1] - ty) < ON_TARGET
                                    for b in local_blobs(fr, tx, ty))))
            flash = max((d for d in near if d[4] >= FLASH), key=lambda d: d[4], default=None)
            flash_xy = None if flash is None else (flash[0], flash[1])
            if tid not in state:
                state[tid] = modes_for((tx, ty))
            for mode, sd, obj in state[tid]:
                if mode == "tbd":
                    obj.step(det.get(k, []))
                else:
                    obj.step(fr, (tx, ty), flash_xy, det.get(k, []))
                ps = obj.pos[-1]
                on.setdefault((tid, mode, sd), []).append(
                    ps is not None and np.hypot(ps[0] - tx, ps[1] - ty) < ON_TARGET)
                fl.setdefault((tid, mode, sd), []).append(flash is not None)

    os.makedirs(REPORTS, exist_ok=True)
    # ---------------- E1 / E2 per track
    e1 = []
    for tid in truth:
        cs = [c for _, c in conf[tid]]
        v = [b for _, b in vis[tid]]
        row = dict(track=tid, n=len(cs), size_px=size[tid])
        for th in THRS:
            det_ok = [c >= th for c in cs]
            row[f"cov_{th}"] = float(np.mean(det_ok))
            row[f"hole_{th}"] = holes(det_ok)
            row[f"run_{th}"] = tbd_longest(det_ok)
            frag, gap, seen = 0, COAST + 1, False
            for hit in det_ok:
                if hit:
                    if gap > COAST:
                        frag += 1
                    gap = 0
                else:
                    gap += 1
            row[f"frag_{th}"] = frag
            row[f"flashes_{th}"] = int(sum(det_ok))
        row["vis_all"] = float(np.mean(v))
        in_hole = [b for b, c in zip(v, cs) if c < 0.1]
        row["vis_in_holes"] = float(np.mean(in_hole)) if in_hole else float("nan")
        row["n_holes_frames"] = len(in_hole)
        row["vis_hole"] = holes(v)
        e1.append(row)
    with open(os.path.join(REPORTS, "e1_e2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(e1[0])); w.writeheader(); w.writerows(e1)

    # ---------------- E3 per track x mode x seed
    e3 = []
    for tid, lst in state.items():
        for mode, sd, obj in lst:
            o = np.array(on[(tid, mode, sd)], bool)
            s = np.array(obj.series, float)
            held = ~np.isnan(s[:, 0])
            run = best = 0
            for h in held:
                run = run + 1 if h else 0
                best = max(best, run)
            fired, correct, s0 = decide(obj.series, o)
            f = np.array(fl[(tid, mode, sd)], bool)
            flash_attr = float(o[f].mean()) if f.any() else float("nan")
            e3.append(dict(track=tid, size_px=size[tid], mode=mode, seed=sd if sd is not None else -1,
                           n=len(o), on_target=float(o.mean()), held=float(held.mean()),
                           longest_run=best, flashes=int(f.sum()), flash_attr=flash_attr, decided=fired, correct=correct, decision_frame=s0 + WIN if fired else -1))
    from common import save
    save({(tid, mode, sd): (list(obj.series), list(on[(tid, mode, sd)]))
          for tid, lst in state.items() for mode, sd, obj in lst},
         os.path.join(CACHE, "e3_series.pkl"))
    with open(os.path.join(REPORTS, "e3.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(e3[0])); w.writeheader(); w.writerows(e3)

    # ---------------- summary
    L = [f"evaluation set: {len(truth)} tracks (Midway, far objects 4-12 px), "
         f"{sum(r['n'] for r in e1)} frames", ""]
    L.append("E1. classifier confidence at the true position (full frame, imgsz 640, gate 45 px)")
    L.append(f"{'size, px':>9s} {'tracks':>6s} | " + " | ".join(
        f"cov@{th} med  hole med  win48" for th in THRS))
    for lo, hi in SIZE_BINS + ((4, 12),):
        rs = [r for r in e1 if lo <= r["size_px"] <= hi]
        if not rs:
            continue
        cells = []
        for th in THRS:
            cells.append(f"{100 * np.median([r[f'cov_{th}'] for r in rs]):8.0f}% "
                         f"{np.median([r[f'hole_{th}'] for r in rs]):9.0f} "
                         f"{sum(r[f'run_{th}'] >= WIN for r in rs):3d}/{len(rs):<3d}")
        L.append(f"{lo:>4d}-{hi:<4d} {len(rs):6d} | " + " | ".join(cells))
    L.append("  cov = share of frames with a detection >= threshold; hole = longest run of frames without;")
    L.append("  win48 = tracks with >= 48 consecutive detected frames (a gap-free series for the spectral test)")
    fr25 = [r["frag_0.25"] for r in e1 if r["flashes_0.25"] > 0]
    L.append(f"  classifier flashes (conf >= 0.25) per track: median {np.median([r['flashes_0.25'] for r in e1]):.0f}; "
             f"tracking-by-detection splits them into median {np.median(fr25):.0f} fragments "
             f"(max {max(fr25)}) -- each fragment is a new object for the system")
    L.append("")
    L.append("E2. visibility in the attention window (threshold from a 120x120 window)")
    for lo, hi in SIZE_BINS + ((4, 12),):
        rs = [r for r in e1 if lo <= r["size_px"] <= hi]
        if not rs:
            continue
        ih = [r["vis_in_holes"] for r in rs if r["n_holes_frames"] > 0]
        L.append(f"{lo:>4d}-{hi:<4d}: all frames {100 * np.median([r['vis_all'] for r in rs]):.0f}% (median), "
                 f"inside detector holes @0.1 {100 * np.median(ih):.0f}%, "
                 f"longest invisible run median {np.median([r['vis_hole'] for r in rs]):.0f}, "
                 f"max {max(r['vis_hole'] for r in rs)}")
    L.append("")
    L.append("E3. closed-loop holding (all 50 tracks; ext modes averaged over 5 cue-noise seeds)")
    L.append(f"{'mode':>18s} | {'on target':>9s} | {'longest run med':>15s} | {'decided':>9s} | {'correct':>9s} | {'wrong obj':>9s}")
    for mode in ("tbd", "att_none", "att_contrast_self", "att_self", "att_self_oracle", "att_ext0.2Hz", "att_ext1Hz"):
        rs = [r for r in e3 if r["mode"] == mode]
        nt = len(rs)
        dec = sum(r["decided"] for r in rs); cor = sum(r["correct"] for r in rs)
        L.append(f"{mode:>18s} | {100 * np.mean([r['on_target'] for r in rs]):8.0f}% | "
                 f"{np.median([r['longest_run'] for r in rs]):15.0f} | "
                 f"{100 * dec / nt:8.0f}% | {100 * cor / nt:8.0f}% | {100 * (dec - cor) / nt:8.0f}%")
    L.append("  share of classifier flashes attributed to the held target (on target at the flash frame):")
    for mode in ("tbd", "att_none", "att_self", "att_self_oracle", "att_ext0.2Hz"):
        rs = [r for r in e3 if r["mode"] == mode and not np.isnan(r["flash_attr"])]
        L.append(f"    {mode:>14s}: {100 * np.mean([r['flash_attr'] for r in rs]):.0f}% (mean over tracks)")
    L.append("  on target = share of frames where the held point is < 25 px from the true one;")
    L.append("  decided = a gap-free 48-frame window of the held series fires the spectral test;")
    L.append("  correct = that window is >= 50 % on target; wrong obj = decided on something else")
    L.append("")
    L.append("E3 by size (att_self | att_ext0.2Hz | att_none): decided correctly")
    for lo, hi in SIZE_BINS:
        cells = []
        for mode in ("att_self", "att_self_oracle", "att_ext0.2Hz", "att_none"):
            rs = [r for r in e3 if r["mode"] == mode and lo <= r["size_px"] <= hi]
            if rs:
                cells.append(f"{mode} {100 * np.mean([r['correct'] for r in rs]):.0f}%")
        L.append(f"  {lo}-{hi} px: " + " | ".join(cells))
    text = "\n".join(L)
    print(text)
    open(os.path.join(REPORTS, "summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()

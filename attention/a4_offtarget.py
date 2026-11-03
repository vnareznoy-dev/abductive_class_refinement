#!/usr/bin/env python3
"""What does the attention window hold when it is NOT on the assigned bird?

Re-runs the main Holder (att_none, att_self, att_ext1Hz x 5 seeds) on the 50 Midway tracks,
checks that the held series equal the cached series of the main run, and for every held frame
farther than 25 px from the assigned target looks for another bird: a position of any of the
625 tracks of the clip (reports/midway_tracks.pkl, multi-hypothesis tracking of the silhouette
channel) within 12 px of the held position in that frame, the track itself being farther than
25 px from the target. Frames with no such track are 'untracked object or background'.
Decision runs scored 'on another object' are checked the same way (majority of their frames).
Output: reports/offtarget.txt, reports/offtarget.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, MIDWAY_TRACKS, REPORTS, frames, load   # noqa: E402
import a4_experiments as X                                                # noqa: E402
import a4_rules as RR                                                     # noqa: E402
from a4_attention_fig import same                                         # noqa: E402

NEAR = 12.0


def main():
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    allt = load(MIDWAY_TRACKS)
    by_k = {}
    for tr in allt:
        for p in tr:
            by_k.setdefault(int(p[0]), []).append((p[1], p[2]))
    modes = [("att_none", None), ("att_self", None)] + [("att_ext1Hz", sd) for sd in X.SEEDS]
    hs, by_frame = {}, {}
    for tid, tr in truth.items():
        for k in tr:
            by_frame.setdefault(k, []).append(tid)
    near_rec = {}
    for k, fr in frames(MIDWAY, min(by_frame), max(by_frame)):
        for tid in by_frame.get(k, []):
            tx, ty = truth[tid][k]
            if tid not in hs:
                hs[tid] = [(m, sd, X.Holder((tx, ty), "ext", period=int(X.FPS_MIDWAY), sigma=15.0, seed=sd)
                            if m == "att_ext1Hz" else X.Holder((tx, ty), "self" if m == "att_self" else "none"))
                           for m, sd in modes]
            near = [d for d in det.get(k, []) if np.hypot(d[0] - tx, d[1] - ty) < X.GATE_DET]
            fl = max((d for d in near if d[4] >= X.FLASH), key=lambda d: d[4], default=None)
            for m, sd, h in hs[tid]:
                h.step(fr, (tx, ty), None if fl is None else (fl[0], fl[1]), det.get(k, []))
                p = h.pos[-1]
                if p is None:
                    rec = "none"
                elif np.hypot(p[0] - tx, p[1] - ty) < X.ON_TARGET:
                    rec = "target"
                else:
                    others = [q for q in by_k.get(k, []) if np.hypot(q[0] - tx, q[1] - ty) >= X.ON_TARGET]
                    rec = "other_bird" if any(np.hypot(q[0] - p[0], q[1] - p[1]) < NEAR for q in others) else "untracked"
                near_rec.setdefault((tid, m, sd), []).append(rec)
    B = load(os.path.join(CACHE, "e3_series.pkl"))
    ok = all(same(h.series, B[(tid, m, sd)][0]) for tid, lst in hs.items() for m, sd, h in lst)
    out, L = {"series_identical_to_main_run": ok}, [f"series identical to main run: {ok}", ""]
    for m in ("att_none", "att_self", "att_ext1Hz"):
        recs = [r for (tid, mm, sd), rs in near_rec.items() if mm == m for r in rs]
        held = [r for r in recs if r != "none"]; off = [r for r in held if r != "target"]
        ob = sum(r == "other_bird" for r in off)
        # decisions on another object
        dec_other, dec_other_bird = 0, 0
        for (tid, mm, sd), rs in near_rec.items():
            if mm != m:
                continue
            s, o = B[(tid, mm, sd)]
            f, c = RR.decide(s, o, RR.FPS_BIRD, "full")
            if f and not c:
                dec_other += 1
                a = np.array(s, float); okr = ~np.isnan(a[:, 0]); best = cur = 0; end = -1
                for i, v in enumerate(okr):
                    cur = cur + 1 if v else 0
                    if cur > best:
                        best, end = cur, i
                seg = rs[end - best + 1:end + 1]
                if sum(r == "other_bird" for r in seg) > sum(r == "untracked" for r in seg):
                    dec_other_bird += 1
        out[m] = dict(held_frames=len(held), off_target=len(off), off_other_bird=ob,
                      share_off=len(off) / max(1, len(held)), share_other_bird_of_off=ob / max(1, len(off)),
                      decisions_other=dec_other, decisions_other_bird=dec_other_bird)
        L.append(f"{m:11s} held {len(held):6d}  off target {len(off):6d} ({100 * len(off) / max(1, len(held)):.0f} % of held)  "
                 f"of them near another tracked bird {ob} ({100 * ob / max(1, len(off)):.0f} %)  "
                 f"decisions on another object {dec_other}, of them on another tracked bird {dec_other_bird}")
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "offtarget.txt"), "w").write(t + "\n")
    json.dump(out, open(os.path.join(REPORTS, "offtarget.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

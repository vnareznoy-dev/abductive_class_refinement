#!/usr/bin/env python3
"""E29 (revision of 4b, reviewer 1, other remark 3). Processing time per frame of the identity step.

  Python: one RevHolder(alp) step on 1920x1080 Midway frames (window threshold + silhouettes +
          constraint (2) + choice (3)), over up to 600 frames of 5 tracks in which the holder has a prediction (lost frames excluded); the decoding of the
          video is excluded.
  Prolog: one query explain(H, J) of a4_track_alp.pl after asserting the facts of a frame
          (f_pred, f_cand, f_body), over the logged ALP decisions of cache/e10_alp_sample.pkl.
Single thread, Apple M-series laptop. Output: reports/e29_timing.txt"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, MIDWAY, REPORTS, frames, load   # noqa: E402
import a4_rev_runs as RV                                    # noqa: E402


def main():
    import cv2
    cv2.setNumThreads(1)
    tracks = sorted(load(os.path.join(CACHE, "eval_tracks.pkl")), key=lambda t: -len(t["pts"]))[:5]
    truth = {t["id"]: {int(p[0]): (p[1], p[2]) for p in t["pts"]} for t in tracks}
    f0 = min(min(tr) for tr in truth.values()); f1 = max(max(tr) for tr in truth.values())
    hs, dt = {}, []
    for k, fr in frames(MIDWAY, f0, f1):
        for tid, tr in truth.items():
            if k not in tr:
                continue
            if tid not in hs:
                hs[tid] = RV.RevHolder(tr[k], "none", gate="alp", floor=3.0); continue
            h = hs[tid]; held = h.last is not None
            t0 = time.perf_counter(); h.step(fr, tr[k], None, []); t1 = time.perf_counter()
            if held:
                dt.append(t1 - t0)
        if len(dt) >= 600:
            break
    py = 1000 * np.array(dt)
    from pyswip import Prolog
    pl = Prolog(); pl.consult(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a4_track_alp.pl"))
    sample = load(os.path.join(CACHE, "e10_alp_sample.pkl")); qt = []
    for pred, cands, body, chosen in sample:
        t0 = time.perf_counter()
        for q in ("f_pred(_,_)", "f_cand(_,_,_,_)", "f_body(_)"):
            list(pl.query(f"retractall({q})"))
        pl.assertz(f"f_pred({float(pred[0])},{float(pred[1])})")
        for n, (x, y, sz) in enumerate(cands):
            pl.assertz(f"f_cand({n},{float(x)},{float(y)},{float(sz)})")
        if body is not None:
            pl.assertz(f"f_body({float(body)})")
        list(pl.query("explain(H, _)", maxresult=1))
        qt.append(time.perf_counter() - t0)
    pq = 1000 * np.array(qt)
    L = ["E29: processing time per frame and target, ms (single thread)",
         f"  Python identity step (window, silhouettes, (2), (3)), {len(py)} frames: median {np.median(py):.2f}, p95 {np.percentile(py, 95):.2f}",
         f"  Prolog facts + explain/2, {len(pq)} frames: median {np.median(pq):.2f}, p95 {np.percentile(pq, 95):.2f}",
         f"  frame period: 33.4 ms at 29.97 Hz, 40 ms at 25 Hz"]
    t = "\n".join(L); print(t); open(os.path.join(REPORTS, "e29_timing.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

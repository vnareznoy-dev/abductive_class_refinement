#!/usr/bin/env python3
"""Fig. 2 for the ІММ version v8d: same three typical tracks (a4_figures.pick_examples: per size
bin the track whose coverage at threshold 0.1 is closest to the bin median), green = silhouette
found by the window threshold (3) within 2*S of the true position (S = track median size).
Output: fig/v8d/fig2_conf.png"""
import inspect, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_figs_v5 as V5          # noqa: E402
import a4_figures as F           # noqa: E402
from common import CACHE, MIDWAY, frames, load, local_blobs   # noqa: E402


def series_for_2s(tids):
    tracks = {t["id"]: t for t in load(os.path.join(CACHE, "eval_tracks.pkl"))}
    det = load(os.path.join(CACHE, "det_midway.pkl"))
    truth = {tid: {int(p[0]): p for p in tracks[tid]["pts"]} for tid in tids}
    size = {tid: float(np.median([max(p[3], p[4]) for p in tracks[tid]["pts"]])) for tid in tids}
    res = {tid: [] for tid in tids}
    for k, fr in frames(MIDWAY, 0, max(max(v) for v in truth.values())):
        for tid in tids:
            if k in truth[tid]:
                p = truth[tid][k]
                c = max((d[4] for d in det.get(k, []) if np.hypot(d[0] - p[1], d[1] - p[2]) < 45), default=0.0)
                v = any(np.hypot(b[0] - p[1], b[1] - p[2]) < 2 * size[tid] for b in local_blobs(fr, p[1], p[2]))
                res[tid].append((k, c, v, p))
    return res, tracks


F.series_for = series_for_2s
V5.FIG = os.path.join(V5.FIG, "v8d"); os.makedirs(V5.FIG, exist_ok=True)
src = inspect.getsource(V5.fig2).replace('"силует у вікні уваги"', '"силует виділено порогом (3)"')
exec(src, V5.__dict__)
V5.fig2()
r, _ = series_for_2s(F.pick_examples())
for tid, s in r.items():
    print(tid, "frames", len(s), "visible %.0f %%" % (100 * np.mean([v[2] for v in s])), "c<0.1 %.0f %%" % (100 * np.mean([v[1] < 0.1 for v in s])))

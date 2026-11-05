#!/usr/bin/env python3
"""E19. Table 4 (classifiers x trackers) with the size-relative on-target gate 2*S.

Re-runs the E6 evaluation of a4_multidet.py (same detection caches, same holders) with
step_all patched to store the DISTANCE from the held position to the true position, then
computes the strict measure (longest gap-free run >= 48 frames, >= 50 % of it on target)
for the gate 25 px (must reproduce reports/e6_table.txt / the article) and 2*S.
Classifier-free У0 is taken from E18 (cache/e18_dist.pkl).
Existing caches/reports are not overwritten.
Outputs: cache/e19_dist.pkl, reports/e19_table4_2s.txt
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, REPORTS, load, save    # noqa: E402
import a4_multidet as M                          # noqa: E402
from a4_multidet_table import run_on             # noqa: E402

USE = ("ours", "v8s")


def step_all_dist(hs, fr, txy, dets, on):
    for m, h in hs:
        if m == "tbd":
            h.step(dets)
        elif m == "att_self":
            h.step(fr, txy, None, dets)
        else:
            h.step(fr, txy, dets)
        p = h.pos[-1]
        on.setdefault(m, []).append(float("nan") if p is None else float(np.hypot(p[0] - txy[0], p[1] - txy[1])))


def job(args):
    kind, name = args
    import cv2, torch
    cv2.setNumThreads(1); torch.set_num_threads(1)
    M.step_all = step_all_dist
    M.MODELS = {name: M.MODELS[name]}
    series = {}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        (M.eval_birds if kind == "bird" else M.eval_uav)(series)
    return series


def on_of(ds, gate):
    return [d is not None and d < gate for d in ds]


def main():
    path = os.path.join(CACHE, "e19_dist.pkl")
    if os.path.exists(path):
        R = load(path)
    else:
        R = {}
        jobs = [(k, n) for k in ("bird", "uav") for n in USE]
        with Pool(len(jobs)) as pool:
            for r in pool.imap_unordered(job, jobs):
                R.update(r); print("  job done", len(R), flush=True)
        save(R, path)
    tracks = load(os.path.join(CACHE, "eval_tracks.pkl"))
    size = {t["id"]: float(np.median([max(p[3], p[4]) for p in t["pts"]])) for t in tracks}
    E18 = load(os.path.join(CACHE, "e18_dist.pkl"))
    L = ["E19 Table 4 with on-target gate 25 px / 2S (strict run on target, %)", "",
         f"{'model: mode':22s} | {'bird 640':>11s} {'bird 1920':>11s} | {'uav 6':>9s} {'uav 9':>9s} {'uav 12':>9s}"]

    def cell(items, gate):
        n = len(items); k = sum(run_on(s, on_of(d, gate(key))) for key, (s, d) in items)
        return 100 * k / n if n else float("nan")

    for n in USE:
        for m in ("tbd", "bytetrack", "botsort", "att_self"):
            row = []
            for gname, gb, gu in (("25", lambda key: 25.0, lambda S: 25.0), ("2S", None, None)):
                vals = []
                for r in M.BIRD_RES:
                    it = [(k, v) for k, v in R.items() if k[0] == "bird" and k[2] == (n, r) and k[3] == m]
                    vals.append(cell(it, (lambda key: 25.0) if gname == "25" else (lambda key: 2 * size[key[1]])))
                for S in (6, 9, 12):
                    it = [(k, v) for k, v in R.items() if k[0] == "uav" and k[1][2] == S and k[2] == n and k[3] == m]
                    vals.append(cell(it, (lambda key: 25.0) if gname == "25" else (lambda key, S=S: 2.0 * S)))
                row.append(f"{gname}: " + " ".join(f"{v:4.0f}" for v in vals))
            L.append(f"{n + ': ' + m:22s} | " + "  ||  ".join(row))
    # classifier-free У0 (bird value is resolution-independent)
    for gname in ("25", "2S"):
        b = [(k, v) for k, v in E18["bird"].items() if k[1] == "att_none"]
        vb = 100 * sum(run_on(s, on_of(d, 25.0 if gname == "25" else 2 * size[k[0]])) for k, (s, d) in b) / len(b)
        vu = []
        for S in (6, 9, 12):
            u = [(k, v) for k, v in E18["uav"].items() if k[3] == "att_none" and k[2] == S]
            vu.append(100 * sum(run_on(s, on_of(d, 25.0 if gname == "25" else 2.0 * S)) for k, (s, d) in u) / len(u))
        L.append(f"{'no classifier: att_none':22s} | {gname}: {vb:4.0f} {vb:4.0f} " + " ".join(f"{v:4.0f}" for v in vu))
    t = "\n".join(L); print(t)
    open(os.path.join(REPORTS, "e19_table4_2s.txt"), "w").write(t + "\n")


if __name__ == "__main__":
    main()

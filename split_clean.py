#!/usr/bin/env python3
"""Build a LEAK-FREE train/val split of the birdrone set.

Three defects of the naive random split, all measured on the data:

1. FRAME ADJACENCY. The images are video frames with sequential numbers. A
   random split put 000002 in train and 000003 in val: 97 % of val frames had a
   training frame at +-1, 100 % within +-2. Validation then re-measures the
   training set. (Roboflow's own split had only 8 % adjacency -- the leak was
   introduced by the re-split, not inherited.)

2. AUGMENTATION DUPLICATES. Roboflow emits several augmented copies per source
   image under different hashes: 3361 drone files over 2788 unique stems. Copies
   of one source image must not straddle the split.

3. SOURCE-DOMAIN CONFOUND (cannot be fixed by any split -- declared, not
   repaired). The classes come from disjoint sources: every one of the 3357
   drone files is a numerically-named video frame, every one of the 394 bird
   files comes from a named clip. Global image statistics alone separate the
   classes at AUC 0.73 (saturation) and 0.69 (brightness), so part of any
   bird/drone discrimination this model learns is provenance, not appearance.
   Consequently the measured confusion rate is an OPTIMISTIC bound and must not
   be quoted as the operational error rate for a camera that sees both classes
   in one frame.

The split here is by SOURCE CLIP, the correct unit:
  * birds  -- the clip is recoverable from the name (`singleBirdinsky012` ->
              `singleBirdinsky#`), so whole clips go to one side;
  * drones -- only a single global frame numbering exists, so contiguous blocks
              of frames serve as pseudo-clips.
"""
from __future__ import annotations

import os, re, glob, shutil, collections, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "rf/birdrone")
DST = os.path.join(HERE, "rf/ds3")
BLOCK = 100          # frames per pseudo-clip for the numerically-named drones
VAL_FRAC = 0.20
SEED = 7


def stem_of(fname):
    b = os.path.basename(fname)
    return b.split("_jpg")[0] if "_jpg" in b else b.split(".rf.")[0]


def classes_of(label_path):
    cl = set()
    if os.path.exists(label_path):
        with open(label_path) as fh:
            for ln in fh:
                p = ln.split()
                if p:
                    cl.add(int(p[0]))
    return cl


def group_key(stem, cls):
    """The source clip a frame came from."""
    if cls == 1:                                  # drone: global numbering
        m = re.match(r"^(\d+)$", stem)
        n = int(m.group(1)) if m else 0
        return f"drone_blk_{n // BLOCK:04d}"
    pat = re.sub(r"\d+", "#", stem)                # bird: named clip
    return f"bird_{pat}"


def main():
    rng = random.Random(SEED)
    # ---- collect, dedupe by stem ------------------------------------------
    by_stem = {}
    for sub in ("train", "valid"):
        for im in glob.glob(os.path.join(SRC, sub, "images", "*")):
            lb = os.path.join(SRC, sub, "labels",
                              os.path.splitext(os.path.basename(im))[0] + ".txt")
            cl = classes_of(lb)
            if len(cl) != 1:
                continue                          # skip empty / mixed
            st = stem_of(im)
            by_stem.setdefault(st, (im, lb, cl.pop()))   # first copy wins
    print(f"унікальних вихідних зображень: {len(by_stem)} "
          f"(з {sum(1 for s in ('train','valid') for _ in glob.glob(os.path.join(SRC,s,'images','*')))} файлів)")

    groups = collections.defaultdict(list)
    for st, (im, lb, c) in by_stem.items():
        groups[group_key(st, c)].append((im, lb, c))

    b_groups = {g: v for g, v in groups.items() if g.startswith("bird_")}
    d_groups = {g: v for g, v in groups.items() if g.startswith("drone_")}
    print(f"груп-джерел: птахи {len(b_groups)}, дрони {len(d_groups)}")
    top = sorted(b_groups.items(), key=lambda x: -len(x[1]))[:5]
    print("найбільші птахові кліпи: " +
          ", ".join(f"{g[5:]}={len(v)}" for g, v in top))

    # ---- assign WHOLE groups, per class, to hit the val fraction ----------
    def assign(gs):
        names = sorted(gs, key=lambda g: -len(gs[g]))
        total = sum(len(gs[g]) for g in names)
        target = VAL_FRAC * total
        val, got = [], 0
        # largest-first would put one huge clip in val; shuffle the mid-sized
        # ones so val draws from SEVERAL clips instead of one
        pool = names[1:] if len(names) > 3 else names[:]
        rng.shuffle(pool)
        for g in pool:
            if got >= target:
                break
            val.append(g); got += len(gs[g])
        return set(val), got, total

    v_b, gb, tb = assign(b_groups)
    v_d, gd, td = assign(d_groups)
    print(f"у val пішло: птахи {gb}/{tb} кліпів={len(v_b)}, "
          f"дрони {gd}/{td} блоків={len(v_d)}")

    # ---- write ------------------------------------------------------------
    for part in ("train", "val"):
        for kind in ("images", "labels"):
            d = os.path.join(DST, part, kind)
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

    counts = {("train", 0): 0, ("train", 1): 0, ("val", 0): 0, ("val", 1): 0}
    for g, items in groups.items():
        part = "val" if (g in v_b or g in v_d) else "train"
        for im, lb, c in items:
            base = os.path.basename(im)
            shutil.copy(im, os.path.join(DST, part, "images", base))
            tgt = os.path.join(DST, part, "labels",
                               os.path.splitext(base)[0] + ".txt")
            if os.path.exists(lb):
                shutil.copy(lb, tgt)
            counts[(part, c)] += 1

    with open(os.path.join(DST, "data.yaml"), "w") as fh:
        fh.write(f"train: {DST}/train/images\n")
        fh.write(f"val: {DST}/val/images\n")
        fh.write("nc: 2\nnames: ['bird','drone']\n")

    print()
    print(f"{'частина':>8s} | {'bird':>6s} {'drone':>7s} {'разом':>7s}")
    print("-" * 34)
    for part in ("train", "val"):
        b, d = counts[(part, 0)], counts[(part, 1)]
        print(f"{part:>8s} | {b:6d} {d:7d} {b+d:7d}")

    # ---- verify the leak is gone -----------------------------------------
    def dnums(part):
        out = []
        for f in os.listdir(os.path.join(DST, part, "images")):
            m = re.match(r"^(\d+)$", stem_of(f))
            if m:
                out.append(int(m.group(1)))
        return out
    T, V = set(dnums("train")), dnums("val")
    for dd in (1, 2, 3):
        near = sum(1 for v in V
                   if any((v + o) in T for o in range(-dd, dd + 1) if o))
        print(f"перевірка: val-кадрів із train-сусідом у ±{dd}: "
              f"{near}/{len(V)} = {100*near/max(1,len(V)):.1f} %")
    tr_st = {stem_of(f) for f in os.listdir(os.path.join(DST, "train", "images"))}
    va_st = {stem_of(f) for f in os.listdir(os.path.join(DST, "val", "images"))}
    print(f"перевірка: спільних stem між train і val: {len(tr_st & va_st)}")
    print(f"\nготово: {DST}/data.yaml")


if __name__ == "__main__":
    main()

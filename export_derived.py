#!/usr/bin/env python3
"""Export the DERIVED data behind the article's silhouette results as plain CSV.

The raw videos and images are third-party and are not redistributed (sources and
licenses: README.md). What is exported here is what this repository computed from
them, so that the spectral results can be re-checked without downloading anything:

  data_derived/bird_tracks.csv     per-frame boxes of the three verified bird tracks
                                   (cormorant, heron, Midway key track #338)
  data_derived/midway_all_tracks.csv  all 625 Midway tracks of flap_midway.py
  data_derived/uav_runs.csv        per-frame annotated UAV boxes of the 9 continuous
                                   birdrone runs (box sizes from the birdrone v10
                                   annotations, CC BY 4.0, Roboflow user fatin-zamri)
  data_derived/decoy_tracks.csv    the false detector track over water (protocol P4)

Columns: source, track, frame (frame index in the source; for the heron and the UAV
runs: index within the track / the birdrone frame number), cx, cy, w, h in pixels.
Needs the raw data in place only for the cormorant and heron tracks (vid2/).
"""
import csv, os, pickle, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "data_derived")


def write(name, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "track", "frame", "cx", "cy", "w", "h"])
        for r in rows:
            w.writerow([r[0], r[1], int(r[2])] + [round(float(v), 3) for v in r[3:]])
    print(f"{name}: {len(rows)} rows")


def main():
    os.makedirs(OUT, exist_ok=True)
    import flap_real as FR
    import flap_specificity as FS
    import compare_sota as CS
    import flap_midway_keytrack as MK

    birds = []
    seq, _ = FR.track(FR.VID, 1240, 1800)
    for p in FR.runs(seq)[0]:
        birds.append(("Kormoran_(Tiefflug).webm", "cormorant", p[0], p[1], p[2], p[3], p[4]))
    for name, _fps, boxes in CS.bird_tracks():
        if "heron" in name or "stork" in name:
            for i, b in enumerate(boxes):
                birds.append(("Kamenitý_rybník,_letící_pták.ogv", "heron", i, *b))
    midway = pickle.load(open(os.path.join(HERE, "reports/midway_tracks.pkl"), "rb"))
    _k, key = MK.pick_track(midway)
    for p in key:
        birds.append(("Midway_Atoll_-_Bird_Sightings_-_Mar-Apr_2015.webm", "midway_338", *list(p)[:5]))
    write("bird_tracks.csv", birds)

    write("midway_all_tracks.csv",
          [("Midway_Atoll_-_Bird_Sightings_-_Mar-Apr_2015.webm", k, *p)
           for k, t in enumerate(midway) for p in t])

    uav = []
    for i, r in enumerate(FS.collect_drone()):
        for j, (w, h) in enumerate(r):
            uav.append(("birdrone v10", f"run_{i+1}", j, float("nan"), float("nan"), w, h))
    write("uav_runs.csv", uav)

    dec = pickle.load(open(os.path.join(HERE, "reports/decoy_tracks.pkl"), "rb"))
    write("decoy_tracks.csv", [("Kormoran_(Tiefflug).webm", name, i, *b)
                               for name, _fps, boxes in dec for i, b in enumerate(boxes)])


if __name__ == "__main__":
    main()

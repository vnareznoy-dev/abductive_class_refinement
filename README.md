# Abductive class refinement for airborne objects

Reference implementation for the article *"Abductive refinement of airborne
object class based on the groundability principle of perceptual facts"*
(Volodymyr Nareznoy, 2026).

A logical fact about an observed object is asserted **only when it is
measurable** by the available sensor at the available range. On top of this
groundability principle, an abductive logic program (ALP) combines the
defeasible appearance evidence of a neural detector with hard kinematic and
silhouette facts, and for every verdict returns either a proof chain or a
justified abstention (`unknown`).

**What is published here is code.** The datasets and videos are third-party
and are downloaded separately from the sources listed below; they are not
redistributed. The version of this repository that corresponds to the article
is the tag `v1.0-article3`.

## Reproducibility map

Status: **verified** = re-run from this repository on the listed data and the
numbers match the article; **pending** = code published, numbers being
re-measured; **not published** = the experiment is described in the article but
its materials are not in this repository.

| Result in the article | Script | Data | Saved report | Status |
|---|---|---|---|---|
| Theory ⟨P, A, IC⟩, 16 cases | `test_theory.py` | none | stdout | verified |
| Measurability tests T1–T3 | `features.py` | none (synthetic) | stdout | verified |
| Revision asymmetry, degradation vs trained classifier (Table 2, Fig. 3) | `evaluate.py` | CTU-MRS + synthetic birds | stdout | verified |
| Wing-beat on real video, cormorant 11×8 px, 4.69 Hz, 2.0× | `flap_real.py` | Kormoran video | stdout | verified |
| Specificity: 0 of 27 signals, 9 runs, 2738 frames (3 measures) | `flap_specificity.py` | birdrone v10 | stdout | verified |
| Specificity with the 4 measures used on birds: 0 of 36 signals; bird vs UAV spectra (Fig. 5) | `fig5_spectra.py` | Kormoran video, birdrone v10 | `reports/fig5_spectra.txt`, `reports/fig5_spectra.png` | verified |
| Protocols P1, P2 (leave-one-track-out, cross-domain) | `compare_sota.py` | Kormoran + Kamenitý videos, birdrone v10 | `reports/compare_sota_P1P4.log` | verified |
| Protocol P3 (box-noise stress) | `compare_sota.py` | same | `reports/compare_sota_P1P2P3.log` | verified |
| Protocol P4 (decoy tracks) | `alp_on_failures.py` → `compare_sota.py` | Kormoran video + trained detector | `reports/alp_on_failures.txt`, `reports/compare_sota_P1P4.log` | verified |
| Smallest silhouette, Midway 5×6 px | `flap_midway.py`, `flap_midway_keytrack.py` | Midway video | `reports/midway_run.log`, `reports/midway_keytrack.txt`, `reports/midway_track338_sheet.jpg` | verified |
| Detector quality, label confusion, confidence of wrong labels | `train2.py`, `eval_detector.py` | birdrone v10 (split below) | `reports/eval_detector.txt` | verified (retrained; see note) |
| Detector outside its distribution (false detections on water), ALP on them | `alp_on_failures.py` | Kormoran video + trained detector | `reports/alp_on_failures.txt` | verified |
| Can the detector hold a small object (scale sweep) | `detector_smallobj.py` | Kormoran video + trained detector | `reports/detector_smallobj.txt` | verified |
| Threshold sensitivity sweep | `sensitivity.py` | CTU-MRS + synthetic birds | stdout | code published, not re-verified |

Detector retraining (`train2.py`, `seed=7, deterministic=True`) is not
bit-identical across hardware and library versions, so detector-dependent
numbers can differ from the article within training variance; the
re-measurement of 2026-09 is summarised in `reports/REMEASURE.md`.

## Data sources

| Material | Source | Version / identifiers used |
|---|---|---|
| CTU-MRS target trajectories | Dataset of *"Towards Safe Mid-Air Drone Interception"*, arXiv:2405.13542 (IROS 2024). Archive: https://nasmrs.fel.cvut.cz/index.php/s/s1KTlblLKUtZTeJ/download | Folder `iros2024-towards-interception-trajectories/main_dataset/csv/`, all 100 files `resampled_trajectory_data_1.csv` … `_100.csv`, used with stride 2 |
| birdrone images | Roboflow Universe, workspace `fatin-zamri`, project `birdrone-dataset`: https://universe.roboflow.com/fatin-zamri/birdrone-dataset/dataset/10 — CC BY 4.0 | Version 10 (2024-04-18), YOLOv8 export of 2024-05-20: 3755 files (train 982 + valid 2773, no test split) |
| Cormorant video | https://commons.wikimedia.org/wiki/File:Kormoran_(Tiefflug).webm — author Rhetos, **CC BY-SA 4.0** | Frames 1240–1800; verified track 1351–1398 |
| Heron video | https://commons.wikimedia.org/wiki/File:Kamenit%C3%BD_rybn%C3%ADk,_let%C3%ADc%C3%AD_pt%C3%A1k.ogv — author Juandev, **CC BY-SA 3.0** | Whole clip, camera-motion compensated track |
| Midway video | https://commons.wikimedia.org/wiki/File:Midway_Atoll_-_Bird_Sightings_-_Mar-Apr_2015.webm — Forest Starr and Kim Starr, **CC BY 2.0** | SHA-1 9269da20ac3391d8ba45379f4bd9b5543d5e0ce3; upper 30 % of the frame; key track 1159–1215 |
| Synthetic bird trajectories | `birds.py` (this repository, tag `v1.0-article3`) | 90 trajectories: flapping, soaring, gliding — **synthetic, not observed data** |

Place the files as: `vid2/Kormoran_Tiefflug.webm`,
`vid2/Kamenitý_rybník_letící_pták.ogv`, `vid3/midway.webm`, `rf/birdrone/`
(Roboflow export unpacked), and set `CTU_MRS_DIR` to the folder that contains
`data/csv/`.

## Derived data (`data_derived/`, made by `export_derived.py`)

The raw videos and images are not redistributed, but everything this repository
measured from them is published as plain CSV (columns: source, track, frame, cx, cy,
w, h in pixels), so the silhouette results can be re-checked without the raw data:

| File | Content |
|---|---|
| `bird_tracks.csv` | per-frame boxes of the three verified bird tracks: cormorant (frames 1351–1398), heron (89 frames, camera-motion compensated), Midway key track #338 (frames 1159–1215) |
| `midway_all_tracks.csv` | all 625 tracks found by `flap_midway.py` in the Midway video |
| `uav_runs.csv` | annotated UAV box sizes of the 9 continuous birdrone runs (2738 frames; derived from the birdrone v10 annotations, CC BY 4.0, Roboflow user fatin-zamri) |
| `decoy_tracks.csv` | the false detector track over water used in protocol P4 |

## Train/validation split of birdrone (`split_clean.py`)

What the script does, exactly:

* takes the export's `train` and `valid` folders;
* drops images whose label file is empty or contains both classes (4 empty,
  0 mixed in v10);
* removes repeated images **by the identifier in the file name** (Roboflow
  augmented copies share it; the first copy is kept). Images are unique by that
  identifier, not by verified content;
* groups images and assigns **whole groups** to one side (20 % validation,
  seed 7):
  * birds — by clip identifiers recovered from the file names;
  * drones — by consecutive blocks of 100 frame numbers (**pseudo-clips**).
    Whether different blocks come from different source videos is **not
    established**.

Result: 3153 images = 2468 training (264 with birds) + 685 validation
(101 with birds). Frame adjacency, defined as the share of validation drone
frames (denominator 584) that have a training frame at ±1 in the frame
numbering: 2.1 % (97 % for a random split). It grows with the distance
(±10: 20.5 %) because blocks are contiguous, so adjacency near zero does not
prove that the two sides are independent. The classes also come from disjoint
sources (global image statistics separate them at AUC 0.73), so any detector
metric on this split is an optimistic bound.

## Quick start (no data needed)

```bash
pip install -r requirements.txt      # + SWI-Prolog: apt install swi-prolog / brew install swi-prolog
python3 test_theory.py               # 16/16 theory cases
python3 features.py                  # measurability gates on synthetic birds
```

## License

MIT (code). Datasets and videos keep their authors' licenses listed above;
figures made from frames of the CC BY-SA videos are adaptations and inherit
CC BY-SA.

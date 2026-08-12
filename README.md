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

Key measured results reproduced by this code:

* wing-beat detection from an 8–12 px silhouette on real video
  (zero false firings over 2,738 continuous UAV frames);
* the asymmetry of label-revision directions under grounding error;
* graceful degradation where a trained classifier falls below the raw
  detector, and abstention on decoy tracks where trained baselines
  confidently answer "bird".

## Layout

| file | role |
|---|---|
| `air_objects.pl` | the abductive theory ⟨P, A, IC⟩ (SWI-Prolog) |
| `test_theory.py` | 16 unit cases for the theory (no data needed) |
| `features.py` | observation pipeline + the three measurability tests T1–T3 |
| `birds.py` | synthetic bird-track generator (flapping / soaring / gliding) |
| `pipeline.py` | detector → gated association → angular facts → ALP |
| `flap_real.py` | the silhouette channel on real video, end-to-end to a verdict |
| `flap_specificity.py` | specificity test of the wing-beat criterion on UAV frames |
| `track_multi.py` | multi-hypothesis tracking with camera-motion compensation |
| `split_clean.py` | leak-free clip-separated split of the birdrone image set |
| `train2.py` | two-class (bird/drone) YOLOv8s training |
| `evaluate.py` | comparison vs detector / threshold rule / trained classifier |
| `sensitivity.py` | threshold sensitivity sweep with feature-support reporting |
| `compare_sota.py` | feature-level proxies of published approaches, protocols P1–P4 |
| `detector_smallobj.py` | can the detector hold a small object? scale sweep |

## Quick start (no data needed)

```bash
pip install -r requirements.txt      # + SWI-Prolog: apt install swi-prolog / brew install swi-prolog
python3 test_theory.py               # 16/16 theory cases
python3 features.py                  # measurability gates on synthetic birds
```

## Real-video experiments

The two validation clips are open Wikimedia Commons files:

* `Kormoran (Tiefflug).webm` — Mathe-AC Lernwerkstatt Aachen, CC BY 4.0
* `Kamenitý rybník, letící pták.ogv` — CC BY 4.0

Download them into `vid2/` (keep the file names), then:

```bash
python3 flap_real.py                 # silhouette → f_flap → ALP verdict
```

## Datasets

* **birdrone** (images, CC BY 4.0) — Roboflow Universe; place under
  `rf/birdrone/`, then `python3 split_clean.py` builds the clip-separated
  split and `python3 flap_specificity.py` runs the specificity test.
* **CTU-MRS pursuit-evasion trajectories** — https://mrs.fel.cvut.cz;
  place the resampled CSV files (columns `time, pos_x, pos_y, pos_z,
  vel_x, vel_y, vel_z`) under `$CTU_MRS_DIR/data/csv/` and set
  `CTU_MRS_DIR` for `evaluate.py` / `sensitivity.py` / `compare_sota.py`.
* **SOD4SB** (MVA2023 challenge) is cited in the article for context and is
  not required by this code.

## License

MIT (code). Dataset and video licenses belong to their authors (CC BY 4.0 as
listed above).

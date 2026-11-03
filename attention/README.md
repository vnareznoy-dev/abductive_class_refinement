# Attention experiments (Article 4)

Holding a 4–12 px aerial target between sparse classifier responses: the holding
threshold (local statistics of a 120×120 attention window around the prediction)
is separated from the decision threshold (the spectral wing-beat criterion of
`flap_real.band_test`, never lowered).

Requires the data of the parent repository (`vid3/midway.webm`,
`vid2/Kormoran_Tiefflug.webm`, `rf/birdrone/`), the detector trained by
`train2.py`, and `reports/midway_tracks.pkl` from `flap_midway.py`.

| Step | Script | Output |
|---|---|---|
| Evaluation set: 50 tracks, 4–12 px, len ≥ 96, straightness ≥ 0.8, visual check | `a4_select.py` | `cache/eval_tracks.pkl`, `reports/select.txt`, `reports/sheets/` |
| Detector boxes on every frame (imgsz 640, CPU) | `a4_detcache.py` | `cache/det_midway.pkl`, `cache/det_kormoran.pkl` |
| E1 classifier coverage, E2 window visibility, E3 six holding strategies + oracle self-cue (upper bound) | `a4_experiments.py` | `reports/e1_e2.csv`, `reports/e3.csv`, `reports/summary.txt` |
| E4 false decisions on water false tracks | `a4_specificity.py` | `reports/e4_specificity.txt` |
| E4b attention on drone runs and background | `a4_specificity_drones.py` | `reports/e4b_specificity_drones.txt` |
| Figures of the article | `a4_figures.py` | `fig/` |

Run in this order from this folder: `python3 a4_select.py && python3 a4_detcache.py && python3 a4_experiments.py && python3 a4_specificity.py && python3 a4_figures.py`.
Numbers of the article tables: `reports/tables.txt`.

## Draft v3 additions (2026-10-03)

| Step | Script | Output |
|---|---|---|
| UAVs: DUT Anti-UAV (Zhao et al., IEEE T-ITS 2022, Apache-2.0), open-sky segments by a pre-registered rule, equivalent size 6/9/12 px by frame reduction | `a4_drones.py` (data in `../dut/`, not redistributed) | `reports/e5_drones.csv`, `reports/e5_summary.txt` |
| Detector at native 1920 px on Midway | `a4_detcache_native.py` | `reports/e1b_native.txt`, `reports/e3b_tbd_native.txt` |
| Decision rules (primary: one test on the longest continuous held run, fixed before evaluation; sensitivity: any / pair / majority of windows) | `a4_rules.py`, `a4_diag_fp.py` | `reports/rules.txt`, `reports/diag_fp.txt`, `reports/diag_truth_rules.txt`, `reports/fps_sensitivity.txt` |
| Figures v3 | `a4_flowchart.py`, `a4_gallery.py`, `a4_figs_v3.py` | `fig/` |
| Attention at work (Fig. 7, 8 of v4), main `Holder` with per-frame trace; series checked against `e3_series.pkl` / `e5_series.pkl` | `a4_attention_fig.py` | `fig/fig_attn_*.png`, `supp/attn_bird.gif`, `supp/attn_uav.gif` |

Key finding: the "any 48-frame window fires" rule is a multiple test; it labels
49 % (25 Hz) of held UAV series as birds. One test on the continuous series gives
0 of 432 (25 Hz), 1.6 % (30 Hz).

## Articles 4a and 4b (version v1.0-article4, 2026-11)

Article 4 was split into two papers. Every number of both papers comes from the
reports in `attention/reports/` (re-run logs `*.log` are included).

**4a. Separating the holding and decision thresholds for tracking small aerial targets**

| Result | Script | Report |
|---|---|---|
| Table 1: classifier coverage and continuous series (640 and 1920 px) | `a4_experiments.py`, `a4_detcache_native.py` | `reports/e1_e2.csv`, `reports/e1b_native.txt` |
| Table 2: target visibility, window vs full-frame threshold | `a4_experiments.py` | `reports/e1_e2.csv`, `reports/summary.txt` |
| Table 3: holding strategies on 50 bird tracks (strict metric) | `a4_experiments.py`, `a4_strict.py` | `reports/e3.csv`, `reports/final_numbers.txt` |
| Table 4: UAV segments 6/9/12 px, false "bird" decisions | `a4_drones.py`, `a4_strict.py` | `reports/e5_drones.csv`, `reports/e5_summary.txt` |
| Table 5: decision rules (one test vs any / pair / majority of windows), 25 and 30 Hz | `a4_rules.py`, `a4_diag_fp.py` | `reports/rules.txt`, `reports/fps_sensitivity.txt` |
| Table 6: three classifiers × SZV / ByteTrack / BoT-SORT / attention | `a4_multidet.py`, `a4_multidet_table.py` | `reports/e6_multidet.csv`, `reports/e6_table.txt` |
| Water false tracks (specificity) | `a4_specificity.py` | `reports/e4_specificity.txt` |
| Figures 1–8 | `a4_gallery.py`, `a4_scheme_v5.py`, `a4_flowchart.py`, `a4_figs_v5.py`, `a4_figures.py` | `fig/` |

**4b. Abductive identity of a small aerial target for refining the labels of a neural network classifier**

| Result | Script | Report |
|---|---|---|
| Identity theory (hypotheses same(b) / unseen, constraint: deviation ≤ body length) | `a4_track_alp.pl` (Prolog), mirrored in `a4_pipeline.py` (`Holder(alp=True)`) | Prolog = Python on 1526 frames: `reports/e10_alp.txt` |
| Table 1: frames off target and continuous series, without / with abductive identity | `a4_alp_eval.py` | `reports/e10_alp.txt`, `reports/e10.log` |
| Table 2: classifier labels along the track (TBD, attention, crop, ideal association) | `a4_assist.py` | `reports/e7_summary.txt`, `reports/e7.log` |
| Table 3: classifier vs classifier + abductive program; start from a detection | `a4_v6_runs.py` | `reports/v6_runs.log` |
| Fig. 3: dependence on the label trust threshold | `a4_sens.py` | `reports/e14_sens.txt` |
| Rejected alternative: identity by size and contrast | `a4_identity.py` | `reports/e9_identity.txt` |
| Fig. 1 (identity diagram), Fig. 2 (bird track with identity) | `a4_identity_fig_v7b.py`, `a4_attention_fig.py` (`bird(alp=True)`) | `fig/` |

UAV data: DUT Anti-UAV (Zhao et al., IEEE T-ITS 2022, Apache-2.0), not redistributed;
place it in `dut/`. The scripts run from this folder; paths resolve automatically
when `attention/` sits inside the repository (`common.py`).

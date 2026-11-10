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

**4a, version v1.1 (on-target tolerance r = 2L instead of 25 px; journal «Інформатика та математичні методи в моделюванні»)**

| Result | Script | Report |
|---|---|---|
| Table 2 with r = 2L (window vs full-frame threshold, birds and UAVs) | `a4_gate2s.py` | `reports/e18_gate2s.txt` |
| Table 4 with r = 2L (UAV segments, false "bird" decisions per segment) | `a4_table4_2s.py` | `reports/e19_table4_2s.txt` |
| Visibility of the target in the window (birds; `--uav` for UAVs) | `a4_visibility.py` | `reports/e20_visibility.txt` |
| Bird size: dark mask vs whole bird | `a4_bird_extent.py` | `reports/e21_bird_extent.txt` |
| Working range, L = 32…4 px (coverage, visibility, strict series) and its figure | `a4_range.py`, `a4_fig_range.py` (uses `a4_fig_sizes.py`) | `reports/e22_range.txt` |
| Check on another dataset (Anti-UAV RGB) | `a4_antiuav.py` (uses `a4_clutter.py`) | `reports/e17_antiuav.txt`, `reports/e17_antiuav.csv` |
| Fig. 1 scheme, Fig. 2 mechanism (window vs frame threshold) | `a4_scheme_v8d.py`, `a4_fig_mechanism.py` | `fig/v8d/` |

**4b, version for Applied Aspects of Information Technology (tolerance r = 2L instead of 25 px)**

| Result | Script | Report |
|---|---|---|
| Table 1 and bird decisions with on-target tolerance r = 2L (L = median target size) | `a4_identity_2s.py` | `reports/e23_identity_2s.txt` |
| Table 3, last column: frames on target, reference start / start from a detection, r = 2L | `a4_e13_2s.py` | `reports/e24_e13_2s.txt` |
| Fig. 1: algorithm of holding with abductive identity | `a4_flowchart_en.py` | `fig/en/fig_algo_en.png` |
| Fig. 2: core of the identity theory (printed from `a4_track_alp.pl`) | `a4_prolog_listing_en.py` | `fig/en/fig_prolog_en.png` |
| Fig. 3: identity hypotheses on a real frame | `a4_identity_real.py` (Ukrainian), `a4_identity_real_en.py` | `reports/identity_real_frame.txt` |
| Fig. 4, Fig. 5: bird track with identity; dependence on the trust threshold | `a4_attention_fig_en.py`, `a4_sens_en.py` | `fig/en/` |

The `*_en.py` figure scripts are generated from the Ukrainian ones by `make_en_figs.py` (labels translated,
decimal point); data and drawing code are unchanged.

**4b, version v1.2 (revision after the reviews of Applied Aspects of Information Technology)**

Main change: the integrity constraint of abductive identity is |c_b − x̂_t| ≤ 2·L_t (two body lengths) instead of
one body length. In `a4_track_alp.pl` this is `thr(body_mult, 2.0)`; in the Python twin `a4_experiments.Holder`
the coefficient is set by the environment variable `A4_ALP_MULT` (default 1 keeps the runs E10–E24 of the
first version reproducible; the scripts of the main variant set `A4_ALP_MULT=2` themselves).
The coefficient follows the measured residual of the true target to the prediction (E25, Fig. 2 of the revised
article). The caches in `cache/` are not published; each script rebuilds what it needs from the videos.

| Result (numbering of the revised article) | Script | Report |
|---|---|---|
| Fig. 2: residual "silhouette centre − prediction" in body lengths, limits 1 L_t and 2 L_t | `a4_rev_residual.py` (E25), `a4_fig_residual.py` | `reports/e25_residual.txt`, `reports/fig_residual.txt` |
| Tables 1 and 2: one pass with all association variants (limits 1 and 2 L_t, floors, fixed and statistical gates; `E26_C2SENS=1`: β and K for 2 L_t) | `a4_rev_runs.py` (E26), `a4_rev_tables.py` | `reports/e26_tables.txt`, `reports/e26_run.log`, `reports/e26b_run.log` |
| Tables 3 and 4 (labels along the track; conclusions vs the trust threshold) for the main variant 2 L_t, exact McNemar tests, cluster-bootstrap intervals; the report keeps the numbering of the first version (its Table 2 / Table 3 are Tables 3 / 4 here) | `a4_rev_c2.py` (E32) | `reports/e32_c2.txt` |
| Labels along the track with the column "window with identity" (limit 1 L_t) and its 95 % intervals | `a4_rev_table2.py` (E27), `a4_rev_table2_ci.py` (E27b) | `reports/e27_table2.txt`, `reports/e27_table2_ci.txt` |
| Statistics of the label decisions vs the trust threshold (first version) | `a4_rev_stats.py` (E28) | `reports/e28_stats.txt` |
| Processing time of the identity step | `a4_rev_timing.py` (E29) | `reports/e29_timing.txt` |
| Start from a detection vs reference start on the same units (first version; main variant in E33) | `a4_rev_detstart.py` (E30) | `reports/e30_detstart_subset.txt` |
| Tolerance of constraint (2) to the residual error of platform-motion compensation | `a4_rev_jitter.py` (E31) | `reports/e31_jitter.txt` |
| Table 5: start from a classifier detection, main variant 2 L_t | `a4_e13_c2.py` (E33) | `reports/e33_e13_c2.txt` |
| Cross-check of the Python twin with c = 2 against E26 and against the Prolog theory | `a4_c2_crosscheck.py` (E34) | `reports/e34_c2_crosscheck.txt` |
| Fig. 3: core of the identity theory with `body_mult` (printed from `a4_track_alp.pl`) | `a4_prolog_listing_en.py` | `fig/en/fig_prolog_c2_en.png` |
| Fig. 4: identity hypotheses on a real frame with the limit 2 L_t (track 37, frame 2043) | `a4_identity_real_c2.py` | `reports/identity_real_c2.txt`, `fig/en/fig1_identity_real_c2_en.png` |
| Fig. 5: bird track with identity, 2 L_t | `a4_fig4_c2.py` (reuses `a4_attention_fig.py`, `a4_attention_fig_en.py`) | `fig/en/fig_attn_bird_alp_c2_en.png` |
| Fig. 6: failure example, a neighbouring bird crossing the target (track 31) | `a4_fig_failure_c2.py` | `reports/fig_failure_c2.txt`, `fig/en/fig_failure_c2_en.png` |
| Fig. 7: dependence of the conclusion on the trust threshold, 2 L_t | `a4_fig_sens_c2.py` (E35) | `reports/e35_sens_c2.txt`, `fig/en/fig_sens_c2_en.png` |

Also added: earlier figure scripts of the Ukrainian drafts (`a4_scheme_v7b.py`, `a4_flowchart_v7b.py`,
`a4_fig2_2s.py`, `a4_fig_antiuav.py`, `a4_fig_attn_antiuav.py`).

UAV data: DUT Anti-UAV (Zhao et al., IEEE T-ITS 2022, Apache-2.0), not redistributed;
place it in `dut/`. The scripts run from this folder; paths resolve automatically
when `attention/` sits inside the repository (`common.py`).

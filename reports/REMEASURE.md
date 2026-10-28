# Re-measurement of Article 3 results (2026-09-24/25)

All numbers below were re-run from this repository; reports are in this folder.
Detector retrained with train2.py on rf/ds3 (40 epochs, best = epoch 39;
the article's run stopped at 27 epochs).

| Result | Article v20 | Re-run | Report |
|---|---|---|---|
| Theory cases | 16/16 | 16/16 | stdout of test_theory.py |
| Cormorant wing-beat | 4.69 Hz, 2.0x, 11x8 px | 4.69 Hz, 2.0x, 11.3x8.4 px | stdout of flap_real.py |
| Specificity | 0/27 signals, 9 runs, 2738 frames | same | stdout of flap_specificity.py |
| CTU-MRS: strict accuracy | 0.89/0.83 vs 0.72; 0.83/0.77 vs 0.75; trained 0.90 -> 0.47 | same | stdout of evaluate.py |
| Split | 3153 = 2468 + 685, birds 264/101, adjacency 2.1 % | same (adjacency denominator 584 val drone frames, +-1) | split_clean.py |
| P1 / P2 | 3/3,2/3,3/3 ; all 2/2, 45/45 | same | compare_sota_P1P4.log |
| P3 box noise 3 px | proxies "bird" 2/2, ours unknown | proxies 10/10, ours 0/10 "bird", 10/10 unknown (5 noise seeds); ours starts abstaining at sigma = 1.0 | compare_sota_P1P4.log |
| P4 decoys | proxies "bird" 4/4, ours unknown 4/4; "4 windows from 2 tracks" | same counts; 4 windows from **1** track | compare_sota_P1P4.log |
| Midway smallest silhouette | 5x6 px, 2.9x @ 3.1 Hz, frames 1159-1215 | same track 5x6 px; all 8 windows fire, 2.3-2.5x @ 2.50 Hz; with a crossing bird (frames 1188-1190) interpolated: 2.84x @ 3.12 Hz | midway_keytrack.txt, midway_track338_sheet.jpg |
| Detector mAP50 bird / drone / all | 0.498 / 0.840 / 0.669 (all: from the working log) | 0.504 / 0.786 / 0.645 | eval_detector.txt |
| Drone recall | 0.889 | 0.858 | eval_detector.txt |
| Label error bird->drone | 9/1528 = 0.6 % [0.3; 1.1] | 9/1350 = 0.7 % [0.4; 1.3] | eval_detector.txt |
| Label error drone->bird | 0.0 % [0.0; 0.7] | 0/505 = 0.0 % [0.0; 0.8] | eval_detector.txt |
| Median conf, correct bird | 0.339 (2 % >= 0.5) | 0.339 (5 % >= 0.5) | eval_detector.txt |
| Median conf, WRONG bird->drone | 0.569 (89 % >= 0.5), n = 9 | **0.456 (11 % >= 0.5)**, n = 9 | eval_detector.txt |
| Median conf, correct drone | 0.579 (91 %) | 0.597 (87 %) | eval_detector.txt |
| False detections on water, conf 0.25 | 3.8/frame | 3.8/frame (3.4 labelled drone) | alp_on_failures.txt |
| det_trust 0.5 on water | "drone" 4.7 -> 0.4 per frame | 3.4 -> 1.0 per frame | alp_on_failures.txt |
| ALP on surviving detector tracks | unknown 3 of 3 false tracks | unknown 3 of 3 false tracks; 4th surviving track is the real bird (detector "bird" 0.60) -> accepted bird | alp_on_failures.txt |
| Detector on full frame, 12-px bird | 0/12 at any imgsz | 0/12 at any imgsz | detector_smallobj.txt |
| Centred window ~29 px | 6/12 on bird, all labelled bird | 8/12 on bird (6 bird, 2 drone); also works at 18 px (3/12) and 45 px (5/12) | detector_smallobj.txt |

Changed conclusions:
1. det_trust: "the 0.5 threshold passes most wrong labels (89 %)" does NOT
   reproduce (11 %). What holds: the threshold removes 95 % of CORRECT bird
   labels, and the median confidence of wrong labels is still above that of
   correct bird labels (0.456 vs 0.339) -- on n = 9 errors, i.e. not robust.
2. Detector as verifier: "all labels in the ~29 px window are correct" does not
   reproduce (2 of 8 labelled drone); the working window is wider than 29 px.
3. det_trust out of distribution suppresses 70 % of false "drone" on water, not 90 %.

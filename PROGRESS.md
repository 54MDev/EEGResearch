# Progress ticker

One row per step. A step is only marked DONE after its output exists and was checked.
**Rule for every future step (incl. Part B onward):** add a row here with (code written, output produced,
what was audited, open issues) before moving on. Code files carry their own purpose/workflow docstrings.

Legend: DONE = built, run, checked. PARTIAL = built/run on a sample only. TODO = not started.

| # | Step | Status | Code | Output | Audit result |
|---|------|--------|------|--------|--------------|
| 0a | Locate data | DONE | -- | `DATA.md` | Data at `Downloads\TDBRAIN_Dataset_V3_1_Encr\...`; 19 GB (docs said 99 GB, fixed in CLAUDE.md); files live under `ses-1/eeg/`. |
| 0b | Environment | DONE | -- | `requirements.txt`, `.venv/` | Python 3.14.7 + all CLAUDE.md pins exact; extras: openpyxl, autoreject 0.5.2. Import check passed. |
| 0c | Spec-vs-data audit | DONE | -- | `AUDIT.md` (22 items) | Key: T3/T4 are T7/T8; ECG detector needs `method="correlation"`; EOG/ECG load as eeg type; recordings ~120 s. ECG decision recorded (correlation + z 3.0). |
| 1 | Cohort (74 subjects) | DONE | `build_sample.py` | `sample.csv` | 37 OCD + 37 HC, all ses-1 restEC, 500 Hz, ages within 1.03 y, 8 OCD are UNKNOWN status (as expected). |
| 2 | Part A: per-subject cleaning | DONE | `clean_subject.py` | `logs/part_a_all74.csv` (all 74; earlier 20-subject check in `logs/part_a_sample_run.csv`; raw arrays in git-ignored `cache/`) | All 74 cleaned in 76 s. **0 excluded** (HC 0/37, OCD 0/37). Clean epochs: HC min 53 / mean 58.9; OCD min 42 / mean 57.8 (lowest `sub-88053677`, 42). All 74 kept 30. ICA removed 1-4 comps/subject; comp 0 in 74/74. FastICA convergence issue for `sub-87960265` fixed (see below); 74/74 now converge. |
| 3 | Part B: PLV features (per-montage CAR) | DONE (2 decisions open) | `features.py`, `test_features.py` | `logs/part_b_check.txt` (6 checks, all pass), `logs/part_b_timing.txt` | Feature counts derived from electrode list: 684/144/40/12. CAR recomputed per montage (complex-domain subtraction, proven equal to time-domain CAR). Wrong-reference test shows the leak is real (max PLV diff 0.46 on random data). Real subject: PLV in [0,1], finite at all four sizes. Cost: all 74 subjects x 19 el = 3.5 s, analytic signals for 74 subjects = 675 MB. Open: AUDIT items 23 (mne-connectivity not used) and 24 (3-el CAR inflates PLV). |
| 3b | ED cross-check (Procedure + Materials) | DONE | -- | `AUDIT.md` section E (items 27-39) | 13 items; Item 27 resolved: `max_features = sqrt` (paste lost the formatting). ED edits tracked in `ED_TODO.md` (Sam doing them). Also material: 28 (predict every count), 31/32 (feature table insufficient; upload cleaned epochs), 33 (37/99 GB). |
| 4 | Part C: one-fold loop | DONE | `onefold.py` | `logs/onefold_sub-87961341.json` (full log), `logs/onefold_run.txt` (console) | Held out `sub-87961341` (control). 16 rounds 19->3, prediction at all 17 counts, feature counts 684/612/.../144(9)/40(5)/12(3) correct. In-fold rule structurally enforced (held-out data used only for prediction). Rerun is bit-for-bit reproducible (same removals, identical inner AUCs). |
| 5 | Part D: timing + counts | DONE | `onefold.py` | same JSON, `summary` block | **167.7 s total (166.5 s loop; 1.2 s load). 65 RF fits, 16 SHAP runs, 17 predictions. Did NOT exceed 15 min -> no 10-fold fallback.** 9 of 16 rounds SHAP and LOGO disagreed (rounds 1,4,5,6,8,10,11,15,16). Tie rule used in 1 round (round 10). Time split: LOGO 101 s, retrain 37 s, SHAP 23 s. |
| 6 | Final conclusion | DONE | -- | `CONCLUSION.md` | Summary of numbers, checks, limits, open decisions. |

## Part C/D audit detail (steps 4-5)
- Machine caveat: timing is from this PC (24 CPUs) with `n_jobs=2` to mimic the planned m5.large (2 vCPU). Core speed differs, so **this is not an AWS measurement**; the AWS run is still needed for a defensible number.
- Scale-up (arithmetic on the measured loop, same machine, not a measurement): 74 x 166.5 s = 12,321 s = 3.4 h. This covers ONLY the 74-fold pruning loop, not the controls (random-removal 50 draws/count, 50 label permutations, discarded-electrode check), not repeats of seeds, not bootstrap.
- Design choices to disclose: (a) inner AUC is subject-level; (b) full-model inner AUC is not fitted, because it is common to all 3 candidates and cancels in "least AUC drop" (keeps the fit count at 65); (c) SHAP uses the OCD class, mean |SHAP| over <=300 seeded epochs, summed over features touching each electrode; (d) tie rule applied to every candidate within 0.01 of the best AUC, dropping the lowest-SHAP one (ED wording covers two candidates; with three it is a small generalisation); (e) one inner split per fold, fixed seed.
- What the single-fold numbers do NOT show: one subject's probabilities (a control: 0.50 at 19 electrodes, 0.71 at 3) say nothing about performance; do not quote them as results.
- Observations to check, not conclusions: (1) inner-split AUC is coarse (18 subjects, 9 vs 9) and noisy, which may explain why SHAP and LOGO disagree in 9/16 rounds; (2) inner AUC is 0.8-1.0 at most counts while the held-out control sits near 0.5; the inner split is chosen once and reused, so it can be optimistic; (3) possible batch confound: in the participants file, `Dataset` and `sessTime` are filled for OCD/HC differently (OCD `sessTime` 0/38 present, HC 138/183), so group could correlate with recording batch. Not verified.

## Part A audit detail (step 2)
- Checked against spec: 1-30 Hz zero-phase FIR, 250 Hz, ICA (EOG z>3.0 via `find_bads_eog`; ECG via `find_bads_ecg(method="correlation", measure="zscore", threshold=3.0)` on `Erbs`), 2 s epochs, Autoreject discards (repairs ignored), first 30 chronological, <20 flagged excluded.
- Channels saved: 19 EEG using T7/T8 naming; not re-referenced (CAR is per-montage in Part B).
- Known limits: seed is a single fixed value (42), seed question still open; library prints a harmless "use rng=" deprecation notice; ICA comp 0 removed in all 20, likely blink (not visually verified).
- All-74 result: see step 2 row. Attrition for the write-up: 0 excluded in either group, 74/74 kept 30 epochs.
- Resolved (Sam chose option 2, 2026-11-22): `sub-87960265` hit FastICA's default 1000-iteration limit. `ICA_MAX_ITER` raised to 5000 for all subjects, convergence now recorded per subject (`ica_converged` column in `logs/part_a_all74.csv` and in each cache json). Rerun of all 74: 74/74 converged, and compared with the previous run **no subject's removed components or clean-epoch count changed** (including `sub-87960265`), so earlier results stand. The cutoff is a parameter, not a methodological change, but disclose it in the ED tech notes.
- Not verified: that ICA component 0 is really a blink (no visual check).

## Part B audit detail (step 3)
- Verified: count derivation, PLV sanity (locked 0.85 vs unrelated 0.31), CAR equivalence, leak check, feature-group indexing, real-subject ranges. Two of my own test designs were wrong first (2-channel CAR forces PLV=1; re-centering cancels a prior CAR) and were corrected; the code itself was only refactored (`plv_of_signals` split out), not changed in behaviour.
- The 0.8 threshold in the sanity test was set after seeing 0.85 vs 0.31; the real criterion is the >0.3 gap.
- Not verified: agreement with `mne_connectivity` PLV (AUDIT item 23); PLV validity at band edges.
- Decisions for Sam: (1) disclose numpy PLV vs. test the library; (2) accept the 3-electrode CAR limitation as a stated limitation.

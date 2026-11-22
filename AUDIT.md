# Discrepancy audit (2026-11-21)

Checked CLAUDE.md and onefoldtest.md against the real data (`build_sample.py` -> `sample.csv`),
the installed MNE 1.13.2 API, and the files themselves. "Fix" = what the spec/code should say.

## A. Would break or silently change the pipeline (fix before coding)

1. **Channel names: T3/T4 do not exist.** Both specs list T3, T4. TDBRAIN files use **T7, T8**
   (same electrodes, modern naming). All 74 cohort files have T7/T8, none have T3/T4. P7/P8 are already correct.
   Fix: use `T7`/`T8` in the 19-channel list (and say T7=T3, T8=T4 in the write-up).
2. **ECG detector does not use a z-score by default.** `ica.find_bads_ecg()` defaults to
   `method="ctps"`, `threshold="auto"` (a phase-statistics test), NOT the z-score 3.0 the specs cite.
   `find_bads_eog()` does default to z-score 3.0. Fix: call `find_bads_ecg(raw, ch_name="Erbs",
   method="correlation", measure="zscore", threshold=3.0)` to match the decided rule. 
   **DECIDED (Sam, 2026-11-21): use correlation + z-score 3.0; ED text stays unchanged.**
3. **EOG/ECG channels load as type "eeg".** MNE reads VPVA, VNVB, HPHL, HNHR, Erbs, Mass as EEG, so they
   would silently enter ICA and the common-average reference. Fix: `set_channel_types` (VPVA/VNVB/HPHL/HNHR
   -> eog, Erbs -> ecg, Mass -> emg, Status -> stim) from `channels.tsv`, and pick only the 19 EEG
   channels for ICA fit, CAR and PLV. The 4 EOG + `Erbs` are the real detection channels (no "proxy" needed).
4. **Recordings are only ~120 s.** Every cohort recording is 120-121 s = ~60 two-second epochs.
   The "keep 30 clean epochs, exclude if <20" rule therefore needs >=50% / >=33% of epochs to survive
   Autoreject. Plausible but not guaranteed; the exclusion count could be non-zero. No fix, but the
   attrition must be measured and reported per group (already required by CLAUDE.md).
5. **File path has an `eeg/` folder.** Real path is `sub-X/ses-1/eeg/sub-X_ses-1_task-restEC_eeg.bdf`.
   My earlier DATA.md omitted `eeg/` (fixed now). `build_sample.ec_path()` is the one place that builds it.

## B. Spec ambiguities that need a decision or an explicit choice

6. **Autoreject repairs by default; the spec says "discard".** Autoreject can interpolate bad channels
   instead of dropping epochs. To match "discard artifacted epochs", drop epochs flagged in
   `reject_log.bad_epochs` and do not use the interpolated output. Needs standard_1020 montage set.
   Autoreject is also randomized: pass `random_state` (CLAUDE.md requires fixed seeds).
7. **Epoch set must be identical across montages.** Autoreject/ICA run once on the 19 channels; the CAR and
   PLV are then recomputed per montage on the *same surviving epochs*. If rejection were rerun per montage
   the epoch set would change between 19/9/5/3 and confound the electrode comparison. Implemented this way.
8. **SHAP for a binary RF returns two classes.** The spec says "absolute SHAP values" but not which class.
   For a 2-class RF the two arrays are mirror images, so use class 1 (OCD) and note it. Low risk, but state it.
9. **Prediction at 19 electrodes.** Spec step 7 (predict after retrain) is inside the loop, so the 19-electrode
   baseline prediction is easy to miss. 17 predictions are expected (19..3), not 16.
10. **CAR with 3 electrodes.** After re-referencing to the average of only 3 channels, the 3 signals sum to
    zero (linearly dependent). PLV is still computable, but it is a known weakness worth a limitations line.
11. **Random seed (already flagged open in CLAUDE.md).** Autoreject, SHAP subsampling, inner split and RF all
    need seeds; single seed by default, structured so a seed loop can be added.

## C. Facts in the docs that do not match the data (cosmetic or need correcting)

12. **"26-channel recording".** Files hold 26 EEG + 4 EOG + 1 ECG + 1 EMG + 1 status = 33 channels in MNE
    (`channels.tsv` and the json say 32, omitting Status). The 26 EEG count is right; the 19 come from those.
13. **Sampling rate** is 500 Hz (confirms the 250 Hz downsample is a 2x decimation). Recordings are already
    **referenced to linked mastoids (A1A2)**; CAR is applied on top. Worth one methods sentence.
14. **Dataset size** was ~99 GB / ~37 GB in CLAUDE.md; real is ~19 GB / 15.5 GB. Fixed earlier.
15. **Counts that DO check out:** 37 OCD with a session-1 row (38 unique OCD subjects exist; sub-88051161 has
    only session 2 so is excluded by the session-1 rule); 8 of them `formal_status = UNKNOWN`; 37 matched
    controls found, max within-pair age gap 1.03 y (limit 2); 74 folds; all 74 have restEC at 500 Hz.
    Feature counts 684/144/40/12 and the ~65 fits / 16 SHAP runs arithmetic are correct.
16. **Citation:** CLAUDE.md cites van Dijk et al. (2022); the dataset's own description file says "(2021)".
    The paper appeared online 2021 and in print 2022; confirm which year your ED uses.
17. **Missing files referenced:** CLAUDE.md/onefoldtest.md mention `ED_progress_tracker.md` and
    `ED_writing_notes.md`; neither is in this repo.
18. **`dataset_description.json` is invalid JSON** (missing comma in Authors). Do not parse it programmatically.
19. **Spreadsheet vs folders:** xlsx lists 1303 unique subjects, disk has 1300 folders. None of the 3 missing
    are in the cohort.
20. **Tech stack list omits `autoreject` (0.5.2) and `openpyxl` (3.1.5)**, both now installed and needed.
    `Consent` has trailing whitespace ("YES ") for 2 subjects; harmless, all 74 are consent = yes.
21. **Greedy matching** (youngest OCD first, nearest same-sex age) is not an optimal assignment. It matched
    all 37 within 1.03 y, so it is adequate, but the ED should say "greedy nearest-age matching".
22. **All 74 are `DISC/REP = DISCOVERY`.** No replication-set subjects are used; that is fine, just state it.

## D. Found while building Part B (2026-11-22)

23. **mne-connectivity is pinned in the ED but not used.** Per-epoch PLV (phase consistency over the 2 s
    within one epoch) is computed in `features.py` with numpy (FFT band-limit + analytic signal).
    I did NOT test `mne_connectivity.spectral_connectivity_time` as an alternative, so I cannot say whether it
    would give the same numbers. Decision needed: say in the ED that PLV is a custom numpy implementation
    (and drop/keep the library in the pinned list), or re-implement with the library and compare.
24. **A 3-electrode CAR inflates PLV.** On a real subject mean PLV rises 0.39 (19 el) -> 0.40 (9) -> 0.44 (5) ->
    0.51 (3): after averaging only 3 channels the signals are linearly dependent. This is evidence for
    item 10 and worth a Limitations sentence (one subject only, not a statistical claim).
25. **Band-limiting is a rectangular FFT mask on a 2 s epoch** (0.5 Hz resolution): delta (1-4 Hz) has only 6
    frequency bins, alpha 10, beta 34; edges [lo, hi). Simple and reproducible, but crude versus a filter bank;
    state it in the methods.
26. **Re-centering makes the reference choice idempotent:** applying per-montage CAR after any earlier CAR gives
    identical features, so the correct behaviour cannot be broken by upstream referencing; only skipping the
    per-montage step breaks it (test shows a max PLV difference of 0.46 in that case on random data).

## E. ED "Procedure" + "Materials" vs CLAUDE.md / onefoldtest.md / what is built (2026-11-22)

Sam pasted the ED text on 2026-11-22. Items below are conflicts or gaps, with the evidence. "Decision" = needs Sam.

27. **RESOLVED (Sam, 2026-11-22): `max_features = sqrt(n_features)`; the ED text lost the sqrt formatting when pasted.** (Original note:) ED: "max_features = # of features". CLAUDE.md / onefoldtest.md: `sqrt(n_features)`.
    These are different models (all features per split = bagging; sqrt = standard RF). Not yet coded (Part C), so
    nothing built depends on it. Arithmetic note: at 684 features, sqrt = 26, so "all features" does ~26x more
    split-evaluation work per node (compute only; no timing measured yet).
28. **RESOLVED (Sam): predict at every count, one at a time; Sam will fix the ED wording (see ED_TODO.md).** (Original note:) ED says predict at 9, 5, 3 only; CLAUDE.md says predict at every count 19 -> 3, one electrode at a time
    (16 rounds).** The ED text "repeat until 9, 5, 3 are left" does not say removal is one at a time. Per CLAUDE.md the
    full curve was explicitly approved by the teacher; the ED wording should match (it also affects the fit count).
29. **ICA "eye, muscle, and heart" vs implementation.** MNE's `find_bads_eog` / `find_bads_ecg` target eye and heart
    only; nothing detects muscle components. Muscle is only caught indirectly (Autoreject, 1-30 Hz filter). Either
    soften the ED ("eye and heart; residual muscle handled by epoch rejection") or add a muscle rule (needs a decision).
30. **ED does not name the epoch-rejection method.** "remove segments that have remaining artifacts" - CLAUDE.md
    decided Autoreject (Jas et al., 2017) and Part A uses it. The ED text has no method, threshold or citation.
31. **ED "Apply a common average reference" - needs the per-montage rule.** CAR must be recomputed on the retained
    electrodes at every montage (CLAUDE.md). Verified why: features for a 9-electrode montage are NOT a column subset
    of the 19-electrode table (max abs PLV difference 0.87 on a real subject).
32. **ED cloud plan "EBS holds only feature table" is not sufficient.** Because of 31, Part C cannot run from one
    precomputed 684-feature table; it needs the cleaned epochs (74 subjects = 84 MB, `cache/`) and recomputes
    PLV per montage (3.5 s for all 74 subjects at 19 el). Upload the cleaned epochs (still not raw data, still tiny).
33. **ED Materials: dataset "37GB zipped, 99GB unzipped".** On this machine: zip 15.5 GB, unzipped 19 GB (CLAUDE.md was
    corrected). If the 37/99 figures describe a different package on the Brainclinics page, state which file you
    downloaded; otherwise update to the measured numbers.
34. **ED Materials: mne-connectivity "used for ... PLV calculations".** Not true of the current code (numpy PLV,
    item 23). Also missing from the list: `autoreject` 0.5.2 (used), `openpyxl` 3.1.5 (reads participants xlsx).
35. **Channel names in ED:** T3/T4 are labelled T7/T8 in the files (item 1). Suggest "T3/T4 (T7/T8 in TDBRAIN)".
36. **Inner validation split "one split per fold" vs spec "computed once per round".** With a fixed seed and the same
    training subjects, regenerating the split every round gives the same split, so results are unaffected; only the
    wording differs. Part C will build it once per fold and reuse it (matches the ED).
37. **Decision threshold "set on training subjects only"** - method not specified (ED or CLAUDE.md). Predictions of an
    RF on its own training data are overly confident, so a threshold from them can mislead. Not needed for the
    one-fold probe; needs a decision before the full 74-fold run.
38. **Machine:** ED says preprocessing runs on an "Academies of Loudoun Computer". Part A/B ran on this PC (Windows 11,
    user profile "Sam"). Confirm which machine you will cite.
39. **Cost estimate "$5-$10, ceiling $20"** is still unmeasured; the one-fold timing (Part C/D) is what replaces it.

Checks that agree with the ED: 37 OCD incl. 8 unknown status; 74 subjects; same subjects at every count; matched
within 2 years (max 1.03); 684/144/40/12; 200 trees, depth 10; ICA before epoching; 2 s epochs; <20 clean = excluded
(0 excluded so far, so the n>=30 note is not triggered); three-lowest-SHAP shortlist; tie rule 0.01; 25% inner split.

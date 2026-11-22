# One-Fold Test — Implementation Spec

## Purpose

This is not the final 74-fold run. This is a **timing and feasibility probe**: implement and run exactly ONE LOSO fold, end to end, and measure what it actually costs. The numbers this produces (wall-clock time, total model fits, total SHAP runs) are what the ED's Feasibility & Scope section needs to replace its current guessed figures (~3-8 hours, ~4,800 fits, ~1,150-1,200 SHAP runs for the full 74-fold run) with a measured one-fold number, scaled up.

Read `CLAUDE.md` first for full project context if you haven't already. This file is the concrete step-by-step spec for the script itself.

## What "one fold" means here

Pick ONE subject to hold out. The other 73 subjects (of the 74 confirmed: 37 OCD + 37 matched controls) are this fold's training pool. Run the complete pruning pipeline using only those 73 subjects' data, down to 3 electrodes, predicting the held-out subject at every electrode count along the way. Then stop — do not loop over the other 73 held-out-subject choices. That's the full 74-fold run, a separate and much later task.

## Part A — Per-subject data cleaning (runs once per subject in the training pool, plus once for the held-out subject)

This is deterministic signal processing, not something that "learns" from data, so it's safe to run once per subject rather than being re-derived per fold.

1. **Select recording and map channels.** Use session 1, eyes-closed recordings only. From each subject's 26-channel TDBRAIN recording, extract the 19 standard 10-20 channels: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, P7, P3, Pz, P4, P8, O1, O2.
2. **Band-pass filter.** Apply a 1-30 Hz zero-phase FIR band-pass filter.
3. **Downsample.** Downsample to 250 Hz.
4. **ICA artifact removal — decided.** Run ICA; remove components flagged by MNE's built-in automated detectors, `ica.find_bads_eog()` and `ica.find_bads_ecg()`, using MNE's documented default z-score threshold of 3.0 for a component's correlation with an EOG/ECG-proxy channel. This step must be done blind to group label (OCD vs. control) — don't let the code path branch on label during this step. This rule is automated and deterministic, so "blind to label" is automatically satisfied; no human judgment call is involved.
   - Cite: Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, Article 267. https://doi.org/10.3389/fnins.2013.00267
5. **Segment.** Cut into non-overlapping 2-second epochs.
6. **Discard artifacted epochs — decided.** Use **Autoreject** (not a hand-picked peak-to-peak µV cutoff) to decide which 2-second epochs still show artifact contamination after ICA. Autoreject derives its rejection threshold per subject via cross-validation on that subject's own epochs, so no global or fold-specific µV number needs to be chosen by hand.
   - Cite: Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage, 159*, 417-429. https://doi.org/10.1016/j.neuroimage.2017.06.030
   - In-fold rule note: Autoreject fits its threshold per subject from that subject's own data, not from other subjects, so there's no cross-subject leakage to worry about here — a held-out subject's threshold is never influenced by training subjects or vice versa. This satisfies the ED's in-fold rule by construction.
7. **Common average reference.** Re-reference to the common average. **Do this per montage, not once up front** — see "Reference recomputation" below.
8. **Cap segments — decided.** Keep up to 30 clean 2-second epochs per subject (60 seconds), taking the **first 30 in chronological order** when more than 30 survive (no random draw, no seed needed for this step). If a subject has fewer than 20 clean epochs, exclude them and record this (for a real run; for the one-fold probe, note if this ever triggers).

## Part B — Feature extraction

1. Compute PLV for every electrode pair in each of the delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) bands, per epoch.
2. At 19 electrodes: 171 pairs × 4 bands = 684 features per epoch. Track how this count falls as electrodes are removed (144 at 9, 40 at 5, 12 at 3) — this should fall out naturally from however you index features by electrode pair, not be hardcoded.
3. **Reference recomputation:** when the electrode set shrinks (at each removal round), recompute the common-average reference using only the currently-retained channels before recomputing PLV for that montage. Do not reuse a reference computed on the full 19-channel set — that would leak dropped-electrode information into the retained channels. Note in your output whether you implemented this correctly; it's a known gap the ED revision plan flagged.

## Part C — The one-fold loop itself

1. **Split off the held-out subject.** All of the chosen subject's epochs go to test; nothing from this subject is used anywhere below until the final prediction step.
2. **Train the baseline 19-electrode Random Forest.** 200 trees, max depth 10, `max_features = sqrt(n_features)`, one fixed random seed. Train on all epochs from the 73 training subjects.
3. **Compute SHAP, aggregate per electrode.** Run SHAP on a random (fixed-seed) subsample of up to 300 training epochs. Take absolute SHAP values and sum them across every PLV feature touching each electrode, giving one importance score per electrode.
4. **Shortlist removal candidates.** Take the 3 lowest-scoring electrodes.
5. **Inner LOGO split.** Carve out a fixed, subject-level, group-stratified inner validation split from the 73 training subjects: 25% held out (~18 subjects), one fixed seed, computed once per round and reused for all 3 candidates in that round.
6. **Run LOGO.** For each of the 3 candidates, retrain without that electrode's entire feature group, measure AUC on the inner validation split. Remove whichever candidate costs the least AUC. If this differs from SHAP's lowest-ranked electrode, follow LOGO and log the mismatch (fold id, round number, electrode). Tie rule: if two candidates' AUC drop is within 0.01 of each other, drop the lower-SHAP-ranked one.
7. **Retrain and predict.** Retrain the model on the reduced electrode set (with the per-montage reference/PLV recomputation from Part B). Generate the held-out subject's prediction at this electrode count: average their epoch-level predictions into one subject-level probability. Record it.
8. **Repeat steps 3-7** one electrode at a time until you've gone from 19 down to 3 electrodes — 16 total removal rounds. Record a prediction at **every** electrode count from 19 down to 3, not only at 9/5/3 — this is the full curve the project wants, and the teacher has explicitly OK'd the compute cost for it.

## Part D — What to measure and report (this is the actual deliverable)

Instrument the whole run and report:

1. **Total wall-clock time** for this single fold, start to finish (cleaning time for the subjects involved can be reported separately from the fold-loop time, since cleaning is a one-time cost amortized across all 74 folds in the real run).
2. **Total Random Forest fits** in this one fold. Expect roughly: 1 baseline fit + per round (3 candidate retrains for LOGO + 1 retrain on the reduced set) × 16 rounds ≈ 65 fits. Report the actual count your code produced, not the estimate.
3. **Total SHAP computations**: expect 16 (one per round, before each round's shortlist step). Report the actual count.
4. **Any SHAP/LOGO mismatches** that occurred, with fold/round/electrode logged.
5. **Whether this fold exceeded 15 minutes** — if it did, that triggers the ED's stated fallback (switch from full LOSO to subject-grouped 10-fold CV), and this should be flagged prominently since it changes the project's design, not just its timeline.

## Output format

Produce a single structured log (JSON or CSV, your choice — document which) containing:
- Fold/subject id held out
- Per-round: round number, electrode count before/after, electrode removed, SHAP's lowest-ranked candidate, LOGO's chosen candidate, whether they agreed, inner-split AUC for each of the 3 candidates
- Per-electrode-count: held-out subject's predicted probability
- Summary block: total wall-clock seconds, total RF fits, total SHAP runs, 15-minute-trigger status (yes/no)

This log is what gets pasted into the ED's Feasibility & Scope section (scaled up by 74 for the full-run estimate) and into `ED_progress_tracker.md` / `ED_writing_notes.md` to replace the current guessed compute numbers. When you finish, summarize these numbers clearly in your final message — don't bury them in code comments only.

## Explicit non-goals for this task

- Do not run all 74 folds. One fold only.
- Do not tune hyperparameters differently per electrode count.
- Do not skip ahead to 9/5/3 — record every count from 19 to 3.
- ICA component selection, artifact-epoch rejection, and which-30-segments are now decided (MNE `find_bads_eog`/`find_bads_ecg`, Autoreject, first-30-chronological, respectively) — implement these exactly, don't substitute an alternative. The random-seed question (single seed vs. repeated seeds) is still genuinely open; don't silently decide it — implement single-seed by default and flag it as open in your output.

# CLAUDE.md

This file orients a Claude Code agent working on Sam Moen's ISEF-track junior research project. Read this fully before touching any code. For the specific task you've most likely been invoked for — building and timing the one-fold LOSO test — read `onefoldtest.md` next; it has the exact implementation spec.

## Project

**Title:** Optimizing Clinical Viability of EEG Diagnostics: A SHAP-Guided Reduction of Electrode Arrays and Its Effect on OCD Classification Performance

**Research question:** At what minimum electrode count (19, 9, 5, or 3) does a Random Forest classifier using PLV connectivity features, pruned with SHAP-guided recursive feature elimination, lose clinical reliability for OCD vs. healthy-control classification?

**Status:** Experimental Design (ED) Draft 7 has been submitted and scored 42/50 against a rubric target of 45/50. The single highest-value fix is the **Feasibility & Scope** section, which currently relies on guessed compute numbers. The project needs a real, timed, single LOSO fold with an exact model-fit count — that is almost certainly why this repo exists and why you've been invoked. Don't write the full 74-fold pipeline yet unless explicitly asked; the one-fold test is a standalone, separately valuable deliverable.

## Data

- **Dataset:** TDBRAIN V3.1 resting-state EEG (van Dijk et al., 2022), already downloaded locally (~15.5 GB zipped / ~19 GB unzipped). Do not re-download it or commit it to version control.
- **Sample:** 37 OCD subjects (`indication` exactly "OCD", comorbid labels excluded; includes 8 subjects with unknown formal diagnosis status, counted as OCD) matched 1:1 by age (±2 years) and sex to 37 healthy controls. **74 subjects total = 74 LOSO folds.** This number is confirmed — do not reintroduce an earlier 76-fold guess that appears in some older planning docs.
- **Session/condition:** Session 1 only, for every subject (assumed pre-treatment; unverified but stated as an assumption in the ED). Eyes-closed recordings only — this is a fixed, pre-specified condition, not a second independent variable.
- **Channels:** Extract the 19 standard 10-20 electrodes (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, P7, P3, Pz, P4, P8, O1, O2) from each subject's 26-channel recording.

## Core methodology (what "the pipeline" means in this repo)

1. **Preprocessing** (once per subject, deterministic, not fold-dependent): 1-30 Hz zero-phase FIR bandpass → downsample to 250 Hz → ICA removing eye/muscle/heart artifacts (done blind to group label) → non-overlapping 2-second epochs → discard epochs with remaining artifacts → common-average reference (recomputed per electrode montage — see "Known open decisions" below) → keep up to 30 clean epochs (60s) per subject; exclude subjects with fewer than 20 clean epochs and report the excluded count per group.
2. **Feature extraction:** Phase Locking Value (PLV) for every electrode pair, in delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) bands. At 19 electrodes that's 171 pairs × 4 bands = 684 features per epoch. Feature count drops to 144 at 9 electrodes, 40 at 5, 12 at 3.
3. **Model:** Random Forest, fixed hyperparameters across every electrode count — 200 trees, max depth 10, `max_features = sqrt(n_features)`, one fixed random seed. Hyperparameters do NOT get re-tuned as electrodes are removed.
4. **Cross-validation:** Leave-One-Subject-Out (LOSO-CV) as the outer loop — 74 folds, one per subject. **The in-fold rule is non-negotiable:** everything that learns from data (SHAP values, LOGO checks, electrode removal choices, the decision threshold) must use only that fold's training subjects. The held-out subject must never influence any of those choices.
5. **Pruning (runs inside every fold, one electrode at a time, 16 rounds, 19→3):** train → compute SHAP on a ≤300-epoch subsample of training data → sum `|SHAP|` across every PLV feature touching each electrode to get one score per electrode → shortlist the bottom 3 electrodes → run LOGO (retrain without each candidate's full feature group) on a fixed, subject-level, group-stratified 25%-of-training-subjects inner split (~18 subjects, same split reused for every candidate within a round) → remove whichever candidate costs the least AUC, following LOGO even when it disagrees with SHAP's ranking (log the mismatch: fold, round, electrode) → tie rule: if two candidates' AUC drop differs by less than 0.01, drop the lower-SHAP-ranked one → retrain on the reduced set and repeat. Predict the held-out subject at **every** electrode count along the way (not just 9/5/3 checkpoints) — the teacher explicitly asked for this full curve since compute budget allows it, and it's already the intended design.
6. **Scoring:** Train on epochs, score per subject — average a held-out subject's epoch-level predictions into one subject-level probability. Decision threshold for sensitivity/specificity is set on training subjects only. Pool held-out probabilities across all 74 folds at each electrode count to compute AUC-ROC (primary metric, threshold-free), sensitivity, and specificity. 95% CIs via subject-level bootstrap (2000 resamples) — never epoch-level.
7. **Controls:** reference baseline = full 19-electrode model (not a "positive control" — rename if you see that term). Random-removal baseline (50 draws per count). Label-permutation test (≥50 shuffles, 19-electrode baseline only). Discarded-electrode robustness check (train on only the pruned-away electrodes; not a negative control). A LOGO-only comparison (no SHAP shortlisting) is planned but not yet implemented — ask before assuming its exact form.

## Decisions made (implement these exactly — do not substitute a different method)

These three were open as of 2026-11-21 and are now decided. Implement them as stated; don't silently pick an alternative even if it seems equivalent.

- **ICA component selection: use MNE's built-in automated detectors**, `ica.find_bads_eog()` and `ica.find_bads_ecg()`, with MNE's documented default z-score threshold of 3.0 for flagging a component's correlation with an EOG/ECG-proxy channel. This replaces "subjective judgment, done blind to label" with a deterministic, automated, citable rule.
  - **Why this over ICLabel or a hand-tuned threshold:** it's MNE's own recommended standard practice, needs no extra trained classifier dependency, and the threshold is a stated, documented number rather than something tuned by eye — which is exactly what the ED rubric asked for (automated criteria with stated thresholds).
  - **Cite:** Gramfort, A., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, Article 267. https://doi.org/10.3389/fnins.2013.00267 (MNE-Python itself); and the MNE-Python documentation for `find_bads_eog`/`find_bads_ecg` for the z-score default.
- **Artifact-rejection threshold for discarding bad epochs: use Autoreject**, not a hand-picked peak-to-peak µV cutoff. Autoreject derives the rejection threshold per subject via cross-validation on the subject's own data, rather than relying on an arbitrary global number.
  - **Why:** it's peer-reviewed, widely used in the EEG preprocessing literature, and its entire purpose is to remove exactly the kind of guessed-threshold problem the ED revision plan flagged. The threshold is learned from data, not picked by a person or by an AI, so it's defensible as "the data decided this, via a cited published method."
  - **Cite:** Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017). Autoreject: Automated artifact rejection for MEG and EEG data. *NeuroImage, 159*, 417-429. https://doi.org/10.1016/j.neuroimage.2017.06.030
  - Implementation note: Autoreject operates per-subject and is typically run on the subject's own epochs (not dependent on group label), so it's compatible with the blind-to-label requirement without extra handling. Still respect the in-fold rule: don't let a held-out subject's Autoreject threshold be influenced by anything from the training subjects, and vice versa — each subject gets its own independently-fit threshold regardless of which fold it's currently playing training vs. test role in.
- **Which 30 segments are kept, when more than 30 clean segments survive: keep the first 30 in chronological order.** No random draw, no seed needed for this step.
  - **Why (reasoned choice, not a literature claim):** a random draw would introduce an extra researcher degree of freedom (which seed? why that seed?) with no clear benefit over the simplest deterministic rule. Taking the first 30 is trivially reproducible by anyone re-running the code and avoids any appearance of cherry-picking which segments to keep.

## Still open — do not silently invent an answer

- **Random seed dependence.** Not yet decided whether one fixed seed is sufficient for the main run, or whether it should repeat with multiple seeds (e.g., 3-5) and report mean ± SD at each electrode count. Bouthillier et al. (2021), "Accounting for Variance in Machine Learning Benchmarks" (arXiv:2103.03098), found that a single fixed seed can meaningfully bias reported ML results and recommends reporting across multiple seeds — relevant context if this gets decided, but the decision itself hasn't been made. If your task touches this, implement single-seed by default (matching the current ED text) but structure the code so adding a seed loop later is a small change, and flag this as still-open in your output rather than deciding it yourself.
- **Common-average reference timing.** This one is a fix, not an open question: the reference must be recomputed separately for each electrode montage (19, 9, 5, 3) at the time that montage is evaluated — never computed once on the full 19 channels and reused for smaller montages, since that would leak dropped-electrode signal into the retained channels through the reference. Implement this; don't treat it as optional.

## Tech stack (pinned versions from the ED)

Python 3.14.7 · MNE-Python 1.13.2 · mne-connectivity 0.9.0 · scikit-learn 1.9.1 · SHAP 0.52.0 · NumPy 2.5.3 · pandas 3.0.6 · matplotlib 3.11.2 · Git/GitHub for version control.

## Compute and budget context

- Local preprocessing happens on the school laptop; only the final feature table (not the raw ~19 GB dataset) gets uploaded to cloud compute.
- Planned cloud instance: AWS EC2 m5.large (fixed-CPU, ~$0.096/hr) in us-east-1. Estimated full-run cost $5-$10 with reruns; billing alert at $20.
- **The school has allotted ~$100 for this project, and unused budget doesn't roll over** — so err toward actually running and timing things rather than estimating, and toward the full one-electrode-at-a-time curve (not skipped checkpoints), since there's headroom to spend.
- Fallback trigger: if one fold takes more than 15 minutes on the lean hyperparameter setup, the design switches to subject-grouped 10-fold CV instead of full LOSO, and this must be stated as a deviation if triggered.

## Code conventions for this project

- This is a human-subjects-adjacent, publicly-reviewed (ISEF-track, mentor-reviewed) research project — write readable, commented code a non-expert mentor could follow, not just working code.
- Never silently change a methodological decision listed above. If something here seems wrong or infeasible, say so and ask — don't quietly substitute a different approach.
- Log everything that could matter for the write-up: per-fold timing, per-round electrode removed, SHAP-vs-LOGO agreement/disagreement, total fit/SHAP-run counts. These logs are themselves a deliverable (they're what the ED's Feasibility section needs).
- Fixed seeds everywhere required by the design above; don't introduce nondeterminism without flagging it.
- Do not fabricate results, performance numbers, or timing estimates. If something hasn't been run, say it hasn't been run.

# One-fold test: conclusion (2026-11-22)

**Bottom line:** the one-fold test ran end to end, finished in **167.7 s** (well under the 15-minute limit), used **65 Random Forest fits** and **16 SHAP runs**, and made **17 held-out predictions** (every count, 19 down to 3). The 10-fold fallback is **not triggered**. This is a timing and feasibility result, not a performance result.

Held-out subject: `sub-87961341` (a control, first in `sample.csv`). Full log: `logs/onefold_sub-87961341.json`.

## 1. Numbers for the ED's Feasibility & Scope section

| Measure | Result | Where from |
|---|---|---|
| Wall-clock, one fold | 167.7 s total (loop 166.5 s, load 1.2 s) | `logs/onefold_sub-87961341.json` summary |
| Random Forest fits | 65 (1 baseline + 16 x (3 LOGO + 1 retrain)) | same |
| SHAP runs | 16 | same |
| Held-out predictions | 17 (19, 18, ... 3 electrodes) | same |
| Exceeded 15 min? | No | same |
| Time split | LOGO 101 s, retrain 37 s, SHAP 23 s | per-round `seconds` in the log |
| SHAP vs LOGO disagreements | 9 of 16 rounds (1, 4, 5, 6, 8, 10, 11, 15, 16) | `mismatches` in the log |
| Tie rule (within 0.01 AUC) used | 1 round (round 10) | `inner_auc` in the log |
| Final 3 electrodes for this fold | Fp2, T7, P3 | `predictions[-1]` |
| Cleaning, all 74 subjects | 76 s total, 0 excluded (HC 0/37, OCD 0/37) | `logs/part_a_all74.csv` |
| PLV features, all 74 subjects at 19 electrodes | 3.5 s | `logs/part_b_timing.txt` |

**Scale-up (arithmetic, not a measurement):** 74 folds x 166.5 s = 12,321 s, about **3.4 hours** on this PC with 2 worker threads. This covers only the 74-fold pruning loop. It excludes the controls (random removal 50 draws per count, 50 label permutations, discarded-electrode check), any repeated seeds, and the bootstrap.

**Caveat on the machine:** timing is from this PC (24 CPUs) using `n_jobs=2` to mimic the planned m5.large (2 vCPU). Core speed differs, so re-time on AWS before quoting an AWS figure.

## 2. What was built (all in this repo)

| Step | Code | Output |
|---|---|---|
| Cohort: 37 OCD + 37 matched controls | `build_sample.py` | `sample.csv` |
| Part A: cleaning (filter, ICA, Autoreject, 30 epochs) | `clean_subject.py` | `logs/part_a_all74.csv` |
| Part B: PLV with per-montage reference | `features.py`, `test_features.py` | `logs/part_b_check.txt`, `logs/part_b_timing.txt` |
| Part C/D: one fold, timed | `onefold.py` | `logs/onefold_sub-87961341.json` |
| Trail of every step | -- | `PROGRESS.md`, `AUDIT.md`, `DATA.md`, `ED_TODO.md` |

## 3. What was checked and held up

- Cohort: 37 OCD with a session-1 row (38 exist; one has only session 2), 8 of them status UNKNOWN, 37 matched controls, largest within-pair age gap 1.03 years.
- Feature counts: 684 / 144 / 40 / 12 at 19 / 9 / 5 / 3 electrodes, derived from the electrode list; 612 at 18 and so on in between.
- Common-average reference recomputed per montage. Verified by test: equals re-referencing before filtering, and differs from reusing the 19-channel reference (max PLV difference 0.46 on random data).
- In-fold rule: the held-out subject's data is used only to make the prediction. SHAP, the inner split, LOGO and electrode removal use the 73 training subjects only.
- Reproducible: a rerun gave the same removal order, identical inner AUCs, and held-out probabilities identical to within 1e-16.
- Cleaning: all 74 ICA fits converge after raising FastICA's iteration limit from 1000 to 5000; no subject's result changed.

## 4. What this does NOT show

- **No performance result.** One held-out control had probability 0.50 at 19 electrodes and 0.71 at 3. A single subject says nothing about AUC, sensitivity or specificity. Do not quote it.
- **No AWS timing and no full-run cost.** The $5-$10 estimate in the ED is still unmeasured.
- **The controls are not built:** random-removal baseline, label permutation, discarded-electrode check, LOGO-only comparison (its exact form is still undefined), decision threshold, bootstrap confidence intervals, selection-frequency analysis.
- **Seed question is still open** (single seed 42 used).

## 5. Things I noticed that are not conclusions

- SHAP and LOGO disagreed in 9 of 16 rounds. The inner validation split has only 18 subjects (9 per group), so its AUC is coarse and noisy. This may explain the disagreement; I did not test it.
- Inner AUC was 0.8 to 1.0 at most electrode counts while the held-out control sat near 0.5. The inner split is chosen once and reused, so it may be optimistic.
- Possible batch confound: in the participants file `sessTime` is filled for 138 of 183 controls and 0 of 38 OCD subjects, and `Dataset` is filled only for OCD. Group may correlate with recording batch. Not verified.
- A 3-electrode common-average reference makes the three signals linearly dependent. On one real subject mean PLV rose from 0.39 (19 electrodes) to 0.51 (3). Worth a Limitations sentence.
- The tie rule in the ED describes two candidates; the code applies it to every candidate within 0.01 of the best AUC.

## 6. Decisions and fixes recorded along the way

| Item | Outcome |
|---|---|
| ECG detector | Correlation + z-score 3.0 (MNE's default for ECG is a different method), so the ED rule is unchanged |
| `max_features` | sqrt(n_features) (the ED paste lost the sqrt) |
| Predictions | At every electrode count, one electrode removed at a time |
| ICA convergence | Iteration limit raised to 5000 for all subjects |
| T3/T4 | Named T7/T8 in the TDBRAIN files |
| Dataset size | 15.5 GB zipped, 19 GB unzipped (docs said 37 / 99 GB; CLAUDE.md corrected, ED still to do) |

## 7. Still to do

- ED edits (Sam): see `ED_TODO.md`. Largest: the cloud plan must upload the **cleaned epochs (84 MB)**, not just a feature table, because features for smaller montages are not a column subset of the 19-electrode table (max PLV difference 0.87).
- Decide: mne-connectivity vs. the numpy PLV; how the decision threshold is set from training subjects only; the exact form of the LOGO-only comparison; single seed vs. repeated seeds.
- Optional checks: the batch confound above; whether ICA component 0 really is the blink component.
- Re-time one fold on the AWS m5.large.
- Nothing in this repo is committed yet.

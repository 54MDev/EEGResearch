# ED edits to make (owner: Sam) -- come back to this

Source: `AUDIT.md` section E. Sam is making these edits himself; nothing here has been changed in the ED by Claude.
Tick a box when the ED is updated.

- [ ] **Cloud plan (item 31, 32):** "EBS holds only the feature table" is not enough. Features for smaller montages are
      not a column subset of the 19-electrode table (max PLV difference 0.87 on a real subject) because the common-average
      reference is recomputed per montage. Upload the **cleaned epochs** instead (74 subjects = 84 MB), recompute PLV per
      montage on the cloud (about 3.5 s for all 74 subjects at 19 electrodes).
- [ ] **Predict at every electrode count (item 28):** state removal is one electrode at a time (16 rounds, 19 -> 3) with a
      held-out prediction at every count, not only 9, 5, 3.
- [ ] **Muscle artifacts (item 29):** MNE's ICA detectors find eye and heart only. Reword "eye, muscle, and heart", or add a
      muscle rule.
- [ ] **Epoch rejection (item 30):** name Autoreject (Jas et al., 2017, NeuroImage 159, 417-429, doi 10.1016/j.neuroimage.2017.06.030),
      learned per subject from that subject's own epochs.
- [ ] **Per-montage reference (item 31):** say the common-average reference is recomputed on the retained electrodes at each montage.
- [ ] **Materials list (items 33, 34):** dataset size (measured 15.5 GB zipped / 19 GB unzipped; list says 37 / 99); mne-connectivity is
      not used for PLV (numpy); add `autoreject` 0.5.2 and `openpyxl` 3.1.5; ICA detector settings (EOG and ECG: z-score 3.0).
- [ ] **Channel names (item 35):** T3/T4 are T7/T8 in the TDBRAIN files.
- [ ] **ICA iteration limit:** FastICA `max_iter` raised from 1000 to 5000 so all 74 fits converge (no subject's result changed).
- [ ] **Still open, needs a decision before the full run (item 37):** how the decision threshold is set from training subjects only.
- [ ] **Confirm (item 38):** which machine is cited as the "Academies of Loudoun Computer" (preprocessing ran on Sam's PC).
- [ ] **Limitations (items 10, 24):** a 3-electrode common-average reference makes the three signals linearly dependent and raised PLV on one subject (0.39 at 19 electrodes, 0.51 at 3).
- [ ] **Feasibility & Scope numbers (from `logs/onefold_sub-87961341.json`):** one fold = 167.7 s (loop 166.5 s) on this PC with 2 worker threads; 65 RF fits; 16 SHAP runs; 17 held-out predictions; under the 15-minute limit, so the 10-fold fallback is not triggered. 74 folds scale to about 3.4 h of loop time (arithmetic, same machine). Re-time on the m5.large before quoting as measured for AWS. Controls (random removal, permutations) are extra.

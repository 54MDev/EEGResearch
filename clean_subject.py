"""clean_subject.py -- Part A of onefoldtest.md: per-subject EEG cleaning.

PURPOSE
    Turns one subject's raw eyes-closed BDF recording into a (n_epochs, 19, 500) array of clean
    2-second epochs (max 30), plus a small JSON log of what was removed. Steps, in order:
    pick 19 channels -> 1-30 Hz zero-phase FIR -> resample to 250 Hz -> ICA (automated EOG/ECG
    component removal) -> 2 s epochs -> Autoreject drops artifacted epochs -> keep first 30.

WORKFLOW CONNECTION
    Reads `sample.csv` (from build_sample.py) for the subject list; writes `cache/<sub>.npy` and
    `cache/<sub>.json`. The next stage (features.py: common-average reference per montage, then
    PLV) reads those arrays. Cleaning is deterministic per subject and uses only that subject's own
    data (ICA and Autoreject are fit per subject), so it cannot leak information between LOSO folds
    and is run ONCE per subject, then reused by every fold. It is blind to group label: the label
    is never read here.

    The saved epochs are deliberately NOT re-referenced: the common-average reference must be
    recomputed per electrode montage (19/9/5/3) later, never once on all 19 channels.

USAGE
    .venv/Scripts/python.exe clean_subject.py [sub-id]     (default: first subject in sample.csv)
"""
import json
import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from mne.preprocessing import ICA
from sklearn.exceptions import ConvergenceWarning

from build_sample import ec_path

# ---- Design constants (CLAUDE.md / onefoldtest.md) ----
# TDBRAIN names T3/T4 as T7/T8 (same electrodes) -- see AUDIT.md item 1.
CH19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
EOG_CH = ["VPVA", "VNVB", "HPHL", "HNHR"]   # real EOG channels in the file
ECG_CH = "Erbs"                             # real ECG channel in the file
LOW, HIGH, SFREQ = 1.0, 30.0, 250
EPOCH_S, MAX_EPOCHS, MIN_EPOCHS = 2.0, 30, 20
Z_THRESH = 3.0                              # decided: z-score 3.0 for both EOG and ECG detectors
ICA_MAX_ITER = 5000                         # raised from MNE's 'auto' (1000): sub-87960265 did not converge at 1000
SEED = 42                                   # single fixed seed (seed question is still open in CLAUDE.md)
CACHE = Path(__file__).with_name("cache")


def load_raw(sub: str) -> mne.io.BaseRaw:
    """Purpose: read the BDF and fix channel types so EOG/ECG/EMG/status channels are not
    treated as EEG (MNE labels them all 'eeg' by default). Keeps only the 19 EEG channels plus
    the EOG/ECG channels needed by the ICA detectors.
    Workflow: first step of cleaning; guarantees ICA, Autoreject and later CAR/PLV only ever see
    the 19 EEG channels."""
    raw = mne.io.read_raw_bdf(ec_path(sub), preload=True, verbose="ERROR")
    raw.set_channel_types({**{c: "eog" for c in EOG_CH}, ECG_CH: "ecg", "Mass": "emg"})
    raw.pick(CH19 + EOG_CH + [ECG_CH])
    raw.set_montage("standard_1020", verbose="ERROR")   # electrode positions (needed by Autoreject)
    return raw


def remove_artifact_components(raw: mne.io.BaseRaw) -> tuple[list[int], bool]:
    """Purpose: ICA on the 19 EEG channels; flag eye and heart components with MNE's automated
    detectors at z-score 3.0 (EOG: find_bads_eog; ECG: find_bads_ecg, correlation + z-score so
    the rule matches the EOG one); remove them in place. Returns (removed component indices, whether FastICA converged).
    Workflow: the decided ICA rule (CLAUDE.md "Decisions made"); no label is used, so it is blind
    to group by construction. Seeded for reproducibility."""
    ica = ICA(n_components=len(CH19), method="fastica", max_iter=ICA_MAX_ITER, random_state=SEED)
    with warnings.catch_warnings(record=True) as w:   # record non-convergence instead of printing it
        warnings.simplefilter("always")
        ica.fit(raw, picks="eeg", verbose="ERROR")
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=EOG_CH, threshold=Z_THRESH, verbose="ERROR")
    ecg_idx, _ = ica.find_bads_ecg(raw, ch_name=ECG_CH, method="correlation",
                                   measure="zscore", threshold=Z_THRESH, verbose="ERROR")
    ica.exclude = sorted(set(eog_idx) | set(ecg_idx))
    ica.apply(raw, verbose="ERROR")
    return ica.exclude, converged


def clean_subject(sub: str) -> dict:
    """Purpose: full Part A pipeline for one subject; saves cache/<sub>.npy and .json and
    returns the log dict (counts, removed ICA components, timing, excluded flag).
    Workflow: called once per subject (74 times in the real run; the one-fold probe needs the 73
    training subjects + 1 held-out). Excluded subjects (<20 clean epochs) must be counted per
    group in the study's attrition report."""
    t0 = time.perf_counter()
    raw = load_raw(sub)
    raw.filter(LOW, HIGH, fir_design="firwin", phase="zero", verbose="ERROR")  # zero-phase FIR
    raw.resample(SFREQ, verbose="ERROR")
    removed, ica_converged = remove_artifact_components(raw)

    raw.pick("eeg")                                          # drop EOG/ECG after ICA is done
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_S, preload=True, verbose="ERROR")
    n_total = len(epochs)

    # Autoreject: per-subject threshold learned from this subject's own epochs. We DISCARD the
    # epochs it flags (spec) and ignore its interpolated/repaired output.
    ar = AutoReject(random_state=SEED, n_jobs=1, verbose=False)
    ar.fit(epochs)
    bad = ar.get_reject_log(epochs).bad_epochs
    clean = epochs[~bad]
    n_clean = len(clean)

    kept = clean[:MAX_EPOCHS]                                # first 30 in chronological order
    data = kept.get_data().astype(np.float32)                # (n, 19, 500), volts
    excluded = n_clean < MIN_EPOCHS

    CACHE.mkdir(exist_ok=True)
    np.save(CACHE / f"{sub}.npy", data)
    log = {"subject": sub, "channels": raw.ch_names, "n_epochs_total": n_total,
           "n_epochs_clean": n_clean, "n_epochs_kept": len(kept), "excluded_lt_20": bool(excluded),
           "ica_removed": [int(i) for i in removed], "ica_converged": ica_converged, "seed": SEED,
           "seconds": round(time.perf_counter() - t0, 1)}
    (CACHE / f"{sub}.json").write_text(json.dumps(log, indent=2))
    return log


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else pd.read_csv(Path(__file__).with_name("sample.csv")).TDBRAIN_ID[0]
    print(json.dumps(clean_subject(sub), indent=2))

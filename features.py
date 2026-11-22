"""features.py -- Part B of onefoldtest.md: PLV connectivity features for any electrode montage.

PURPOSE
    Converts cleaned epochs (n_epochs, 19, 500 samples @ 250 Hz) into Phase Locking Value (PLV)
    features for every electrode pair in four bands (delta 1-4, theta 4-8, alpha 8-13, beta 13-30 Hz),
    for ANY subset of the 19 electrodes. Feature count is derived from the electrode list, never
    hardcoded: k electrodes -> C(k,2) pairs x 4 bands (19->684, 9->144, 5->40, 3->12).

WORKFLOW CONNECTION
    Input: `cache/<sub>.npy` from clean_subject.py. Output: feature matrices consumed by the Random
    Forest / SHAP / LOGO steps (Part C, the one-fold loop). The helper `electrode_feature_groups`
    tells Part C which feature columns "touch" each electrode: that is what SHAP scores are summed
    over to get one score per electrode, and what LOGO removes together when it drops an electrode.

    KEY RULE (CLAUDE.md "Common-average reference timing"): the common-average reference (CAR) must
    be recomputed using ONLY the electrodes in the current montage; reusing a 19-channel CAR would
    leak dropped-electrode signal into the retained channels. Band-pass filtering is linear, so
    subtracting the mean of the retained channels' band-limited complex signals is exactly the same as
    re-referencing first and filtering after. `plv_features` does that subtraction per montage, so the
    expensive step (`analytic_bands`) is done once per subject and the reference is still recomputed
    correctly for every montage. test_features.py proves the equivalence.

    PLV here is the phase consistency WITHIN one 2-second epoch: |mean over time of exp(i*(phi_a - phi_b))|.
    It is computed with numpy (FFT band-limiting + analytic signal) because mne-connectivity's
    across-epoch PLV is a different quantity; see AUDIT.md item 23.
"""
from itertools import combinations

import numpy as np

SFREQ = 250
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


def analytic_bands(data: np.ndarray, sfreq: float = SFREQ) -> np.ndarray:
    """Purpose: band-limit each epoch to each band and return the complex analytic signal.
    data: (n_epochs, n_channels, n_samples) real. Returns (n_epochs, 4, n_channels, n_samples)
    complex64, bands in BANDS order. Band edges are [lo, hi) so no frequency bin is in two bands.
    Workflow: the once-per-subject expensive step. Its output is reference-agnostic (no CAR yet);
    `plv_features` applies the per-montage CAR on top of it."""
    n, c, t = data.shape
    spec = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(t, d=1.0 / sfreq)
    out = np.zeros((n, len(BANDS), c, t), dtype=np.complex64)
    for b, (lo, hi) in enumerate(BANDS.values()):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        full = np.zeros((n, c, t), dtype=np.complex128)
        full[..., idx] = 2.0 * spec[..., idx]          # one-sided spectrum x2 = analytic signal
        out[:, b] = np.fft.ifft(full, axis=-1)
    return out


def feature_names(electrodes: list[str]) -> list[str]:
    """Purpose: the ordered feature labels 'band:ElecA-ElecB' for a montage. Order is band-major,
    then pairs in the order given by `electrodes`. Workflow: defines the column order of
    `plv_features` and is what makes feature counts fall out of the electrode list automatically."""
    return [f"{band}:{a}-{b}" for band in BANDS for a, b in combinations(electrodes, 2)]


def plv_of_signals(z: np.ndarray) -> np.ndarray:
    """Purpose: PLV for every channel pair of an already-referenced analytic signal
    z (n_epochs, 4, k, n_samples). Returns (n_epochs, 4 * k*(k-1)/2), band-major, pairs in
    itertools.combinations order. Workflow: the reference-free core of `plv_features`; split out so
    test_features.py can show what a WRONG (not recomputed) reference would produce."""
    u = z / np.maximum(np.abs(z), 1e-30)               # keep phase only
    plv = np.abs(np.einsum("nbit,nbjt->nbij", u, u.conj())) / u.shape[-1]
    iu = np.triu_indices(z.shape[2], k=1)
    return plv[:, :, iu[0], iu[1]].reshape(plv.shape[0], -1)


def plv_features(analytic: np.ndarray, keep_idx: list[int]) -> np.ndarray:
    """Purpose: PLV features for one montage. keep_idx = row positions of the retained electrodes in
    the 19-channel array (same order as the `electrodes` list passed to `feature_names`).
    Steps: select retained channels -> subtract their mean (CAR over THIS montage only) -> PLV.
    Returns (n_epochs, n_features), values in [0, 1].
    Workflow: called by Part C for each subject at every montage (19, 18, ... 3) after each removal."""
    z = analytic[:, :, keep_idx, :].astype(np.complex128)
    z = z - z.mean(axis=2, keepdims=True)              # per-montage common-average reference
    return plv_of_signals(z)


def electrode_feature_groups(electrodes: list[str]) -> dict[str, np.ndarray]:
    """Purpose: for each electrode, the column indices of every feature whose pair includes it.
    Workflow: Part C sums |SHAP| over these columns for the per-electrode score, and LOGO deletes
    exactly these columns to test an electrode's contribution."""
    names = feature_names(electrodes)
    return {e: np.array([i for i, n in enumerate(names) if e in n.split(":")[1].split("-")])
            for e in electrodes}

"""build_sample.py -- build and audit the 74-subject cohort (37 OCD + 37 matched controls).

PURPOSE
    Turns the TDBRAIN participants spreadsheet into the exact list of subjects used by the
    project, and checks every listed recording against what CLAUDE.md / onefoldtest.md assume
    (file exists, sampling rate, channel names, recording length).

WORKFLOW CONNECTION
    This is Step 0 of the pipeline. Its output, `sample.csv`, is the single source of truth
    that the cleaning step (Part A of onefoldtest.md) and the LOSO loop read from: one row per
    subject = one LOSO fold (74 rows = 74 folds). Nothing here learns from EEG signal, so it
    cannot leak information between folds; it only uses spreadsheet labels, age and sex
    (matching is a study-design step done once, before any cross-validation).

USAGE
    .venv/Scripts/python.exe build_sample.py
"""
from pathlib import Path

import mne
import pandas as pd

# ---- Paths and design constants (see DATA.md and CLAUDE.md "Data") ----
ROOT = Path(r"C:\Users\Sam\Downloads\TDBRAIN_Dataset_V3_1_Encr\TDBRAIN_Dataset_V3_1")
XLSX = Path(r"C:\Users\Sam\Downloads\TDBRAIN_participants_V3.1.xlsx")
OUT = Path(__file__).with_name("sample.csv")
AGE_TOL = 2          # years; CLAUDE.md: age +/-2, exact sex match
SESSION = 1          # CLAUDE.md: session 1 only
# Names the spec lists. TDBRAIN actually uses T7/T8 for T3/T4 (see audit_channels).
SPEC_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
           "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def ec_path(sub: str) -> Path:
    """Purpose: locate a subject's session-1 eyes-closed BDF file.
    Workflow: used by every later stage (cleaning, timing) to find raw data; also by this
    script to confirm the recording exists before the subject is allowed into the cohort."""
    return ROOT / sub / f"ses-{SESSION}" / "eeg" / f"{sub}_ses-{SESSION}_task-restEC_eeg.bdf"


def greedy_match(cases: pd.DataFrame, pool: pd.DataFrame) -> list[tuple[str, str]]:
    """Purpose: pair each OCD subject 1:1 with the closest-age same-sex healthy control.
    Workflow: defines the control half of the cohort so group differences are not just
    age/sex differences. Greedy (youngest OCD first), same rule as the earlier audit script.
    Deterministic: ties broken by pandas index order. Uses labels only, never EEG."""
    pool = pool.copy()
    pairs = []
    for _, r in cases.sort_values("age").iterrows():
        cand = pool[(pool.gender == r.gender) & ((pool.age - r.age).abs() <= AGE_TOL)]
        if len(cand):
            j = (cand.age - r.age).abs().idxmin()
            pairs.append((r.TDBRAIN_ID, pool.loc[j, "TDBRAIN_ID"]))
            pool = pool.drop(j)
    return pairs


def build_cohort() -> pd.DataFrame:
    """Purpose: apply the CLAUDE.md inclusion rules to the spreadsheet.
    Workflow: produces the 74 rows (label 1 = OCD, 0 = control) that become the 74 LOSO folds.
    Rules: indication exactly 'OCD' or 'HEALTHY'; must have a session-1 row; must have a
    session-1 restEC file on disk (controls without one cannot be matched)."""
    df = pd.read_excel(XLSX)
    df = df[df.sessID == SESSION]
    df = df[df.TDBRAIN_ID.map(lambda s: ec_path(s).exists())]
    ocd = df[df.indication == "OCD"]
    hc = df[df.indication == "HEALTHY"]
    pairs = greedy_match(ocd, hc)
    ids = {**{o: 1 for o, _ in pairs}, **{c: 0 for _, c in pairs}}
    out = df[df.TDBRAIN_ID.isin(ids)].copy()
    out["label"] = out.TDBRAIN_ID.map(ids)
    pair_of = {**{o: i for i, (o, _) in enumerate(pairs)}, **{c: i for i, (_, c) in enumerate(pairs)}}
    out["pair_id"] = out.TDBRAIN_ID.map(pair_of)
    return out[["TDBRAIN_ID", "label", "pair_id", "indication", "formal_status", "age", "gender",
                "DISC/REP", "Consent"]].sort_values(["pair_id", "label"]).reset_index(drop=True)


def audit_recordings(cohort: pd.DataFrame) -> pd.DataFrame:
    """Purpose: read each cohort recording's HEADER ONLY (no signal loaded) and record
    sampling rate, channel names and duration, so spec assumptions are checked, not guessed.
    Workflow: tells the cleaning step which channels to pick, whether the 20-clean-epoch
    exclusion rule is likely to trigger (short recordings), and gives the sample.csv columns."""
    rows = []
    for sub in cohort.TDBRAIN_ID:
        raw = mne.io.read_raw_bdf(ec_path(sub), preload=False, verbose="ERROR")
        rows.append({"TDBRAIN_ID": sub, "sfreq": raw.info["sfreq"],
                     "n_channels": len(raw.ch_names), "duration_s": round(raw.times[-1], 1),
                     "has_T7_T8": {"T7", "T8"} <= set(raw.ch_names),
                     "has_T3_T4": {"T3", "T4"} <= set(raw.ch_names),
                     "has_ECG_Erbs": "Erbs" in raw.ch_names,
                     "n_eog": sum(c in raw.ch_names for c in ["VPVA", "VNVB", "HPHL", "HNHR"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cohort = build_cohort()
    info = audit_recordings(cohort)
    full = cohort.merge(info, on="TDBRAIN_ID")
    full.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(full)} subjects; OCD={int(full.label.sum())}, HC={int((1 - full.label).sum())})")
    print("formal_status of OCD arm:", full[full.label == 1].formal_status.value_counts().to_dict())
    print("age gap within pairs (max):", round(full.groupby("pair_id").age.agg(lambda a: a.max() - a.min()).max(), 2))
    for c in ["sfreq", "n_channels", "has_T7_T8", "has_T3_T4", "has_ECG_Erbs", "n_eog", "DISC/REP", "Consent"]:
        print(c, full[c].value_counts(dropna=False).to_dict())
    print("duration_s by group:\n", full.groupby("label").duration_s.describe().round(1).to_string())
    print("recordings shorter than 60 s:", int((full.duration_s < 60).sum()))

# Data location (verified 2026-11-21)

**Do not copy into the repo or commit. Read from these paths.**

| What | Path |
|---|---|
| Dataset root (BIDS-style, unzipped) | `C:\Users\Sam\Downloads\TDBRAIN_Dataset_V3_1_Encr\TDBRAIN_Dataset_V3_1\` |
| Original zip (15.5 GB) | `C:\Users\Sam\Downloads\TDBRAIN_Dataset_V3_1_Encr.zip` |
| Participants spreadsheet | `C:\Users\Sam\Downloads\TDBRAIN_participants_V3.1.xlsx` |
| Participants JSON (data dictionary) | `C:\Users\Sam\Downloads\TDBRAIN_participants_V3.json` |
| Earlier sample audit script (OCD/HC matching, file-name EC/EO check) | `C:\Users\Sam\Downloads\tdbrain_sample_audit.py` |

## Layout
`sub-<id>/ses-1/sub-<id>_ses-1_task-restEC_eeg.bdf` (plus `_channels.tsv`, `_eeg.json`).
Eyes-closed files = `task-restEC`. Other tasks: `restEO`, `oddball` (not used).

## Counts found on disk
- 1300 subject folders; 2940 `.bdf` files total (1391 restEC, 1387 restEO, 162 oddball).
- Unzipped size is ~19 GB (zip 15.5 GB). CLAUDE.md previously said ~99 GB; corrected.
- The 74-subject sample (37 OCD + 37 matched HC) has NOT been re-derived here. Run the audit script against the xlsx to produce the subject ID list.

## Environment (checked)
- Python 3.10.11 (CLAUDE.md pins 3.14.7).
- `mne` imports; `shap` is NOT installed. Other pins (sklearn, mne-connectivity, autoreject) not verified.

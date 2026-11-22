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
`sub-<id>/ses-1/eeg/sub-<id>_ses-1_task-restEC_eeg.bdf` (plus `_channels.tsv`, `_eeg.json` in the same `eeg/` folder).
Eyes-closed files = `task-restEC`. Other tasks: `restEO`, `oddball` (not used).

## Counts found on disk
- 1300 subject folders; 2940 `.bdf` files total (1391 restEC, 1387 restEO, 162 oddball).
- Unzipped size is ~19 GB (zip 15.5 GB). CLAUDE.md previously said ~99 GB; corrected.
- The 74-subject cohort is derived by `build_sample.py` -> `sample.csv` (37 OCD + 37 matched HC, all with ses-1 restEC).
- Full list of spec-vs-data discrepancies: see `AUDIT.md`.

## Project venv (rebuilt 2026-11-21)
`.venv/` in repo (git-ignored), Python 3.14.7 (installed via winget, matches pin). `requirements.txt` is the frozen list.
All pins in CLAUDE.md match exactly. Added (unpinned in ED): openpyxl, autoreject 0.5.2.

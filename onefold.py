"""onefold.py -- Part C + D of onefoldtest.md: ONE LOSO fold of SHAP-guided electrode pruning, timed.

PURPOSE
    Holds out one subject, trains on the other 73, and removes electrodes one at a time (19 -> 3, 16 rounds)
    with SHAP shortlisting + LOGO decision, predicting the held-out subject at every electrode count.
    It also measures what that costs (wall-clock, number of Random Forest fits, number of SHAP runs),
    which replaces the guessed numbers in the ED's Feasibility & Scope section.

WORKFLOW CONNECTION
    Reads `sample.csv` (build_sample.py), cleaned epochs `cache/<sub>.npy` (clean_subject.py) and PLV
    features from features.py (common-average reference recomputed per montage). Writes one structured
    log, `logs/onefold_<subject>.json` (per-round decisions, per-count predictions, summary block), which
    feeds the ED and PROGRESS.md. The real 74-fold run would call `run_fold` once per subject.

    In-fold rule: the held-out subject's data is used ONLY for the prediction at each electrode count.
    SHAP, the inner validation split, LOGO and the electrode removal all use the 73 training subjects only.

USAGE
    .venv/Scripts/python.exe onefold.py [--held-out sub-ID] [--n-jobs 2]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import features as f
from clean_subject import CH19

HERE = Path(__file__).parent
SEED = 42                       # single fixed seed (seed-repeat question is still open in CLAUDE.md)
N_TREES, MAX_DEPTH = 200, 10    # max_features = sqrt(n_features), fixed for every electrode count
SHAP_MAX_EPOCHS = 300
N_CANDIDATES = 3
INNER_VAL_SUBJECTS = 18         # ~25% of 73 training subjects
TIE_MARGIN = 0.01
MIN_ELECTRODES = 3
FIFTEEN_MIN_S = 900


def new_rf(n_jobs: int) -> RandomForestClassifier:
    """Purpose: the one Random Forest configuration used everywhere (never re-tuned per electrode count).
    Workflow: every fit in the fold (baseline, LOGO candidates, per-round retrain) goes through here."""
    return RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, max_features="sqrt",
                                  random_state=SEED, n_jobs=n_jobs)


def make_xy(Z: dict, subs: list[str], keep_idx: list[int], labels: dict):
    """Purpose: stack PLV features (current montage, reference recomputed) for a list of subjects.
    Returns X (epochs x features), y (epoch labels), g (subject id per epoch).
    Workflow: called whenever a model needs training/scoring data for a given electrode set."""
    X, y, g = [], [], []
    for s in subs:
        x = f.plv_features(Z[s], keep_idx)
        X.append(x)
        y += [labels[s]] * len(x)
        g += [s] * len(x)
    return np.vstack(X), np.array(y), np.array(g)


def subject_proba(model, X: np.ndarray, g: np.ndarray) -> dict:
    """Purpose: average epoch-level OCD probabilities into one probability per subject.
    Workflow: the study scores subjects, not epochs (ED: 'averaging their segment predictions')."""
    p = model.predict_proba(X)[:, 1]
    return {s: float(p[g == s].mean()) for s in np.unique(g)}


def subject_auc(model, X, g, labels) -> float:
    """Purpose: subject-level AUC on a validation set. Workflow: the quantity LOGO compares across
    candidates on the inner validation split."""
    sp = subject_proba(model, X, g)
    ids = list(sp)
    return float(roc_auc_score([labels[s] for s in ids], [sp[s] for s in ids]))


def electrode_scores(model, X: np.ndarray, electrodes: list[str], round_no: int) -> dict:
    """Purpose: SHAP importance per electrode. Takes <=300 random training epochs (seeded), computes
    |SHAP| for the OCD class, averages over epochs, then sums over every PLV feature touching each electrode.
    Workflow: step 3 of the fold loop; the 3 lowest scores become the removal shortlist. A feature touches
    two electrodes, so it contributes to both scores."""
    rng = np.random.default_rng(SEED + round_no)
    rows = rng.choice(len(X), size=min(SHAP_MAX_EPOCHS, len(X)), replace=False)
    sv = shap.TreeExplainer(model).shap_values(X[rows], check_additivity=False)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)   # OCD class
    per_feature = np.abs(sv).mean(axis=0)
    groups = f.electrode_feature_groups(electrodes)
    return {e: float(per_feature[idx].sum()) for e, idx in groups.items()}


def choose_removal(candidates: list[str], shap_score: dict, auc: dict) -> str:
    """Purpose: LOGO decision + tie rule. Remove the candidate whose removal leaves the highest inner AUC
    (= least AUC cost; the unmeasured full-model AUC is common to all candidates so it cancels). If other
    candidates are within 0.01 AUC of the best, drop the one with the lowest SHAP score among them.
    Workflow: step 6 of the fold loop; follows LOGO even when it disagrees with SHAP's ranking."""
    best = max(auc.values())
    tied = [c for c in candidates if best - auc[c] < TIE_MARGIN]
    return min(tied, key=lambda c: shap_score[c])


def run_fold(held_out: str, n_jobs: int = 2) -> dict:
    """Purpose: the full one-fold experiment plus instrumentation. Returns the log dict.
    Workflow: Part C (loop) + Part D (measurements). In the real run this is called 74 times."""
    sample = pd.read_csv(HERE / "sample.csv")
    labels = dict(zip(sample.TDBRAIN_ID, sample.label.astype(int)))
    assert held_out in labels
    train_subs = [s for s in sample.TDBRAIN_ID if s != held_out]
    assert held_out not in train_subs and len(train_subs) == 73

    t_start = time.perf_counter()
    Z = {s: f.analytic_bands(np.load(HERE / "cache" / f"{s}.npy").astype(np.float64)) for s in sample.TDBRAIN_ID}
    t_load = time.perf_counter() - t_start

    # One fixed, subject-level, group-stratified inner validation split for the whole fold (ED: one split
    # per fold, fixed seed, reused for every candidate in every round).
    inner_train, inner_val = train_test_split(train_subs, test_size=INNER_VAL_SUBJECTS, random_state=SEED,
                                              stratify=[labels[s] for s in train_subs])

    t_loop = time.perf_counter()
    electrodes = list(CH19)
    n_fits = n_shap = 0
    rounds, predictions, mismatches = [], [], []

    def fit_and_predict(elec: list[str]):
        keep = [CH19.index(e) for e in elec]
        X, y, g = make_xy(Z, train_subs, keep, labels)
        model = new_rf(n_jobs).fit(X, y)
        Xh, _, gh = make_xy(Z, [held_out], keep, labels)
        return model, X, subject_proba(model, Xh, gh)[held_out]

    # Baseline: 19 electrodes
    model, X, p = fit_and_predict(electrodes)
    n_fits += 1
    predictions.append({"n_electrodes": 19, "electrodes": list(electrodes), "n_features": int(X.shape[1]), "held_out_prob": p})

    round_no = 0
    while len(electrodes) > MIN_ELECTRODES:
        round_no += 1
        t_round = time.perf_counter()
        before = len(electrodes)

        t0 = time.perf_counter()
        scores = electrode_scores(model, X, electrodes, round_no)           # SHAP run
        n_shap += 1
        t_shap = time.perf_counter() - t0
        candidates = sorted(scores, key=scores.get)[:N_CANDIDATES]          # lowest summed |SHAP|

        t0 = time.perf_counter()
        auc = {}
        for cand in candidates:                                             # LOGO
            elec_wo = [e for e in electrodes if e != cand]
            keep = [CH19.index(e) for e in elec_wo]
            Xa, ya, ga = make_xy(Z, train_subs, keep, labels)              # reference recomputed without cand
            tr, va = np.isin(ga, inner_train), np.isin(ga, inner_val)
            m = new_rf(n_jobs).fit(Xa[tr], ya[tr])
            n_fits += 1
            auc[cand] = subject_auc(m, Xa[va], ga[va], labels)
        t_logo = time.perf_counter() - t0

        removed = choose_removal(candidates, scores, auc)
        agreed = removed == candidates[0]
        if not agreed:
            mismatches.append({"fold": held_out, "round": round_no, "electrode": removed, "shap_lowest": candidates[0]})
        electrodes = [e for e in electrodes if e != removed]

        t0 = time.perf_counter()
        model, X, p = fit_and_predict(electrodes)                           # retrain on reduced set + predict
        n_fits += 1
        t_retrain = time.perf_counter() - t0
        predictions.append({"n_electrodes": len(electrodes), "electrodes": list(electrodes),
                            "n_features": int(X.shape[1]), "held_out_prob": p})
        rounds.append({"round": round_no, "n_before": before, "n_after": len(electrodes), "removed": removed,
                       "shap_lowest_candidate": candidates[0], "logo_chosen": removed, "agreed": agreed,
                       "candidates": candidates, "shap_scores_candidates": {c: scores[c] for c in candidates},
                       "inner_auc": auc, "seconds": {"shap": round(t_shap, 2), "logo": round(t_logo, 2),
                                                      "retrain_predict": round(t_retrain, 2),
                                                      "round_total": round(time.perf_counter() - t_round, 2)}})
        print(f"round {round_no:2d}: {before}->{len(electrodes)} removed {removed:>3s} "
              f"(SHAP lowest {candidates[0]}, agree={agreed}) {time.perf_counter() - t_round:.1f}s", flush=True)

    loop_s = time.perf_counter() - t_loop
    return {
        "fold_held_out": held_out, "held_out_label_for_reference_only": labels[held_out],
        "config": {"seed": SEED, "n_trees": N_TREES, "max_depth": MAX_DEPTH, "max_features": "sqrt",
                   "n_jobs": n_jobs, "cpu_count_this_machine": os.cpu_count(), "shap_max_epochs": SHAP_MAX_EPOCHS,
                   "inner_val_subjects": INNER_VAL_SUBJECTS, "inner_val_ids": sorted(inner_val),
                   "note": "timing is from THIS machine, not the planned AWS m5.large"},
        "rounds": rounds, "predictions": predictions, "mismatches": mismatches,
        "summary": {"total_wall_clock_s": round(time.perf_counter() - t_start, 1), "loop_s": round(loop_s, 1),
                    "load_and_analytic_s": round(t_load, 1), "total_rf_fits": n_fits, "total_shap_runs": n_shap,
                    "n_rounds": len(rounds), "n_predictions": len(predictions), "n_mismatches": len(mismatches),
                    "exceeded_15_min": loop_s > FIFTEEN_MIN_S},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", default=pd.read_csv(HERE / "sample.csv").TDBRAIN_ID[0])
    ap.add_argument("--n-jobs", type=int, default=2)     # 2 mimics the planned m5.large (2 vCPU)
    a = ap.parse_args()
    log = run_fold(a.held_out, a.n_jobs)
    out = HERE / "logs" / f"onefold_{a.held_out}.json"
    out.write_text(json.dumps(log, indent=2))
    print(json.dumps(log["summary"], indent=2))
    print("wrote", out)

"""
R5 — Threshold recalibration for cross-domain spread transfer.

R4 established the positive half of the thesis (spread signal keeps cross-domain
AUC ~0.79 where content collapses to ~chance) but flagged a recurring failure
mode, first seen in R3: several cross cells have strong ranking (AUC 0.64-0.85)
yet **F1 = 0.000** at the fixed 0.5 decision threshold. R3 and R4 both deferred
the same next step verbatim: "report spread at a recalibrated threshold, not
just 0.5." This script does exactly that, and asks one clean question:

    Is the F1-0.000-at-AUC-0.85 collapse a FIXABLE calibration artifact
    (ranking is usable, only the operating point is wrong), or a genuine
    transfer failure (ranking is not actually usable for a decision)?

We reuse R4's simulator + 28-feature extractor + train/eval protocol VERBATIM
(imported from run_spread_cross_domain.py; no re-tuning) and hold the LEARNED
MODEL fixed. The only thing that varies is how the decision threshold is chosen.
Four policies, ordered from most-honest-deployable to cheating-ceiling:

  1. naive_0.5        — the R4 baseline (reproduces the collapse).
  2. source_prior     — pick the threshold on the SOURCE domain's own scores so
                        the predicted-positive rate matches the known class prior
                        (0.5, balanced), then apply to target. Uses ZERO target
                        labels; only assumes you know the base rate. Deployable.
  3. target_cal       — reserve a SMALL labelled slice of the target domain
                        (N_CAL/class), pick the F1-maximising threshold there,
                        evaluate on the disjoint remainder. Realistic few-label
                        deployment; uses target labels but NOT the test scores.
  4. target_oracle    — F1-maximising threshold chosen on the target TEST labels
                        themselves. This PEEKS at test and is reported ONLY as an
                        upper bound (ceiling), never as a result. It answers "how
                        much ranking power is recoverable in principle."

Honesty framing (Dr. Santos discipline):
  * Cascades are SIMULATED, same as R4. This is mechanism validation about
    CALIBRATION, not an in-the-wild number. AUC is unchanged by thresholding;
    we are only relabelling the same score ranking.
  * target_oracle is explicitly a cheat/ceiling. The deployable claim rests on
    source_prior (no target labels) and target_cal (few target labels).
  * If source_prior alone rescues the F1-0.000 cells, the collapse was a pure
    calibration artifact. If only target_cal / oracle rescue them, then transfer
    needs target supervision — a weaker but still honest positive claim.

Author: Claudio L. Lima
Date: 2026-09-04
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# --- reuse R4's simulator + extractor + domains VERBATIM -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_spread_cross_domain import (  # noqa: E402
    DOMAINS,
    build_domain_dataset,
)


# ---------------------------------------------------------------------------
# Threshold-selection policies. Each returns a scalar threshold in [0,1].
# ---------------------------------------------------------------------------
def _f1max_threshold(proba, y):
    """Threshold on `proba` that maximises F1 against labels `y`.

    Sweeps every candidate cut (midpoints between sorted unique scores, plus the
    extremes) and returns the arg-max. Ties broken toward the lower threshold.
    """
    y = np.asarray(y)
    uniq = np.unique(proba)
    if uniq.size == 1:
        return 0.5
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    cands = np.concatenate([[0.0], mids, [1.0]])
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _prior_threshold(source_proba, prior=0.5):
    """Threshold so predicted-positive rate on the SOURCE equals `prior`.

    For a balanced prior this is the median of the source score distribution.
    Uses no target information whatsoever.
    """
    q = np.quantile(source_proba, 1.0 - prior)
    return float(q)


def _eval_at(proba, y, thr):
    pred = (proba >= thr).astype(int)
    return dict(
        thr=round(float(thr), 4),
        f1=round(f1_score(y, pred, zero_division=0), 3),
        cm=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def main():
    N_TRAIN = 150   # per class (matches R4)
    N_CAL = 25      # per class, reserved from target for target_cal policy
    N_TEST = 60     # per class, the held-out target test (matches R4 val size)
    SEED = 20260904

    print("R5 — Threshold recalibration for cross-domain spread transfer")
    print("=" * 64)

    # Build, per domain: a train set, and an independent eval pool that we split
    # into a small calibration slice + a disjoint test slice.
    train, cal, test = {}, {}, {}
    names = None
    for i, dom in enumerate(DOMAINS):
        Xtr, ytr, names = build_domain_dataset(dom, N_TRAIN, SEED + i)
        Xpool, ypool, _ = build_domain_dataset(dom, N_CAL + N_TEST, SEED + 500 + i)
        # split pool into cal (first N_CAL/class) and test (rest), keeping balance
        pos = np.where(ypool == 1)[0]
        neg = np.where(ypool == 0)[0]
        cal_idx = np.concatenate([pos[:N_CAL], neg[:N_CAL]])
        test_idx = np.concatenate([pos[N_CAL:], neg[N_CAL:]])
        train[dom] = (Xtr, ytr)
        cal[dom] = (Xpool[cal_idx], ypool[cal_idx])
        test[dom] = (Xpool[test_idx], ypool[test_idx])
        print(f"  built {dom}: train {Xtr.shape}, cal {cal[dom][0].shape}, "
              f"test {test[dom][0].shape}")

    domains = list(DOMAINS)
    matrix = {}
    # aggregate F1 per policy over cross cells (the ones that matter)
    agg = {p: {"cross": [], "in": []}
           for p in ["naive_0.5", "source_prior", "target_cal", "target_oracle"]}
    auc_cross, auc_in = [], []

    for tr in domains:
        Xtr, ytr = train[tr]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr), ytr)

        # source scores (for source_prior threshold) — the model on its own train
        src_proba = clf.predict_proba(scaler.transform(Xtr))[:, 1]
        thr_prior = _prior_threshold(src_proba, prior=0.5)
        thr_src_f1 = _f1max_threshold(src_proba, ytr)  # informational

        for te in domains:
            Xte, yte = test[te]
            Xca, yca = cal[te]
            proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
            proba_cal = clf.predict_proba(scaler.transform(Xca))[:, 1]
            try:
                auc = round(roc_auc_score(yte, proba), 3)
            except ValueError:
                auc = float("nan")

            thr_cal = _f1max_threshold(proba_cal, yca)       # few target labels
            thr_oracle = _f1max_threshold(proba, yte)        # cheat / ceiling

            cell = {
                "auc": auc,
                "naive_0.5": _eval_at(proba, yte, 0.5),
                "source_prior": _eval_at(proba, yte, thr_prior),
                "target_cal": _eval_at(proba, yte, thr_cal),
                "target_oracle": _eval_at(proba, yte, thr_oracle),
                "source_f1max_thr": round(thr_src_f1, 4),
            }
            matrix[f"{tr}->{te}"] = cell
            bucket = "in" if tr == te else "cross"
            (auc_in if tr == te else auc_cross).append(auc)
            for p in agg:
                agg[p][bucket].append(cell[p]["f1"])

    # ---- summary ----------------------------------------------------------
    def mean(xs):
        return round(float(np.mean(xs)), 3) if xs else float("nan")

    summary = {
        "mean_auc_in": mean(auc_in),
        "mean_auc_cross": mean(auc_cross),
        "cross_f1_by_policy": {p: mean(agg[p]["cross"]) for p in agg},
        "in_f1_by_policy": {p: mean(agg[p]["in"]) for p in agg},
        "n_cross_collapsed_naive": int(sum(f < 0.05 for f in agg["naive_0.5"]["cross"])),
        "n_cross_collapsed_source_prior": int(
            sum(f < 0.05 for f in agg["source_prior"]["cross"])),
        "n_cross_cells": len(agg["naive_0.5"]["cross"]),
    }

    print("\n--- cross-domain F1 by threshold policy "
          "(AUC is identical across policies) ---")
    print(f"  mean cross-domain AUC (unchanged) : {summary['mean_auc_cross']}")
    for p in ["naive_0.5", "source_prior", "target_cal", "target_oracle"]:
        print(f"  cross F1  [{p:14s}] : {summary['cross_f1_by_policy'][p]}")
    print(f"\n  cross cells with F1<0.05 (collapsed):")
    print(f"    naive 0.5     : {summary['n_cross_collapsed_naive']}"
          f"/{summary['n_cross_cells']}")
    print(f"    source_prior  : {summary['n_cross_collapsed_source_prior']}"
          f"/{summary['n_cross_cells']}")

    print("\n--- per-cell (CROSS cells are where the R3/R4 collapse lived) ---")
    for cell, r in matrix.items():
        s, t = cell.split("->")
        tag = "(in) " if s == t else "CROSS"
        print(f"  {tag} {cell:16s} AUC {r['auc']:.3f} | "
              f"F1 naive {r['naive_0.5']['f1']:.3f} -> "
              f"prior {r['source_prior']['f1']:.3f} | "
              f"cal {r['target_cal']['f1']:.3f} | "
              f"oracle {r['target_oracle']['f1']:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(os.path.join(_HERE, "..", "data",
                                       f"spread_threshold_recal_{ts}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(feature_names=names, matrix=matrix, summary=summary,
                       config=dict(n_train=N_TRAIN, n_cal=N_CAL, n_test=N_TEST,
                                   seed=SEED)),
                  f, indent=2)
    print(f"\nRaw output: {out}")


if __name__ == "__main__":
    main()

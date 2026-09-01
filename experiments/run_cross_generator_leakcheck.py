#!/usr/bin/env python3
"""
R2b — Is the cross-generator AUC INVERSION real, or a shortcut-feature artifact?

R2 (run_cross_generator.py, Sep 1) found something sharper than a generalization
gap: content detectors don't just fail across generators, their ranking INVERTS
(cross-generator AUC ~= 0, not ~0.5). But I flagged a caveat honestly in
RESULTS.md: synthetic_test scores AUC 1.0 in-domain because ~7 of the 29 forensic
features perfectly separate its real/fake classes (a construction artifact —
real=brighter/sharper, fake=diffusion-smoothed). CIFAKE has 0 such features.

The worry: those same opposite-polarity shortcut features could be the ENTIRE
mechanism behind the AUC~=0 inversion. If so, the inversion is an artifact of one
dataset, not evidence about content detection in general.

This script settles it, with zero new data and the identical 29-feature pipeline:

  1. Rank every feature by per-feature separability (|AUC-0.5| direction-free) on
     each training set. Auto-identify "shortcut" features: near-perfect on
     synthetic_test but weak on CIFAKE.
  2. Re-run the full 2x2 cross-generator matrix TWICE:
        (a) ALL 29 features            (reproduces R2)
        (b) shortcut features REMOVED  (the honest stress test)
  3. Verdict:
        - inversion SURVIVES removal  -> it's a real cross-generator phenomenon
        - inversion COLLAPSES to ~0.5 -> R2's headline was a shortcut artifact;
                                         RESULTS.md must be corrected.

Either outcome is a publishable result and a Dr.-Santos-clean one: we don't get
to keep the dramatic number unless it survives its own leak check.

Label convention: fake=1 (positive), real=0. Features imported verbatim from
run_cross_generator.py so R2 and R2b are byte-for-byte comparable.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

# Reuse the EXACT feature extraction + loaders from R2 (no reimplementation drift).
from run_cross_generator import extract_features, load_split, DATA  # noqa: E402

DATASETS = ["cifake", "synthetic_test"]

# Human-readable names for the 29-d feature vector (color9 + radial_fft16 + noise4).
FEATURE_NAMES = (
    [f"{ch}_{stat}" for ch in ("R", "G", "B") for stat in ("mean", "std", "skew")]
    + [f"fft_bin{i:02d}" for i in range(16)]
    + ["resid_mean", "resid_std", "resid_absmean", "resid_abs_p90"]
)


def per_feature_separability(X, y):
    """Direction-free separability of each feature = max(AUC, 1-AUC) in [0.5, 1]."""
    sep = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.allclose(col, col[0]):
            sep[j] = 0.5
            continue
        a = roc_auc_score(y, col)
        sep[j] = max(a, 1.0 - a)
    return sep


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def run_matrix(train, val, cols):
    """2x2 train/test matrix restricted to feature indices `cols`."""
    matrix = {}
    for tr in DATASETS:
        Xtr, ytr = train[tr]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
        model.fit(Xtr[:, cols], ytr)
        matrix[tr] = {}
        for te in DATASETS:
            Xva, yva = val[te]
            res = evaluate(model, Xva[:, cols], yva)
            res["kind"] = "in_domain" if tr == te else "cross_generator"
            matrix[tr][te] = res
    return matrix


def cross_gen_aucs(matrix):
    return [matrix[tr][te]["auc"]
            for tr in DATASETS for te in DATASETS if tr != te]


def print_matrix(title, matrix):
    print(f"\n--- {title} ---")
    for tr in DATASETS:
        for te in DATASETS:
            r = matrix[tr][te]
            tag = "IN-DOMAIN" if tr == te else "CROSS-GEN"
            print(f"  [{tag}] train={tr:14s} test={te:14s} "
                  f"F1 {r['f1']:.4f} | AUC {r['auc']:.4f}")


def main():
    t0 = time.time()
    print("Loading + featurizing all splits (29 forensic features)...")
    train = {d: load_split(d, "train") for d in DATASETS}
    val = {d: load_split(d, "val") for d in DATASETS}
    for d in DATASETS:
        Xtr, ytr = train[d]
        print(f"  {d}: train {Xtr.shape[0]} "
              f"({int((ytr==0).sum())}r/{int((ytr==1).sum())}f)")

    # --- 1. Per-feature separability on each training set --------------------
    sep = {d: per_feature_separability(*train[d]) for d in DATASETS}
    SHORTCUT_HI = 0.98   # "perfectly" separates a set
    WEAK_LO = 0.75       # but is only a weak cue elsewhere
    shortcut_idx = [
        j for j in range(len(FEATURE_NAMES))
        if sep["synthetic_test"][j] >= SHORTCUT_HI and sep["cifake"][j] < WEAK_LO
    ]
    keep_idx = [j for j in range(len(FEATURE_NAMES)) if j not in shortcut_idx]

    print("\n=== Per-feature separability (max(AUC,1-AUC), in-set) ===")
    print(f"{'feature':16s} {'cifake':>8s} {'synth':>8s}   flag")
    for j, name in enumerate(FEATURE_NAMES):
        flag = "SHORTCUT" if j in shortcut_idx else ""
        print(f"{name:16s} {sep['cifake'][j]:8.3f} "
              f"{sep['synthetic_test'][j]:8.3f}   {flag}")
    print(f"\nShortcut features (synth>= {SHORTCUT_HI}, cifake< {WEAK_LO}): "
          f"{len(shortcut_idx)}/29 -> {[FEATURE_NAMES[j] for j in shortcut_idx]}")

    # --- 2. Matrices: all features vs shortcut-removed -----------------------
    m_all = run_matrix(train, val, list(range(29)))
    m_clean = run_matrix(train, val, keep_idx)
    print_matrix("ALL 29 FEATURES (reproduces R2)", m_all)
    print_matrix(f"SHORTCUT REMOVED ({len(keep_idx)} features)", m_clean)

    # --- 3. Verdict ----------------------------------------------------------
    cross_all = cross_gen_aucs(m_all)
    cross_clean = cross_gen_aucs(m_clean)
    mean_all, mean_clean = float(np.mean(cross_all)), float(np.mean(cross_clean))

    def classify(mean_auc):
        if mean_auc <= 0.35:
            return "INVERSION (ranking anti-correlated)"
        if mean_auc >= 0.65:
            return "TRANSFER (detector generalizes)"
        return "CHANCE (no cross-generator signal)"

    verdict_all, verdict_clean = classify(mean_all), classify(mean_clean)
    inversion_survives = mean_clean <= 0.35

    print("\n=== VERDICT ===")
    print(f"  mean cross-gen AUC, ALL features : {mean_all:.4f}  -> {verdict_all}")
    print(f"  mean cross-gen AUC, CLEAN        : {mean_clean:.4f}  -> {verdict_clean}")
    if inversion_survives:
        print("  => Inversion SURVIVES shortcut removal: it is a real "
              "cross-generator phenomenon, NOT a synthetic_test artifact.")
    elif mean_clean >= 0.65:
        print("  => Inversion COLLAPSES to transfer once shortcuts are removed: "
              "R2's headline was a shortcut artifact. Correct RESULTS.md.")
    else:
        print("  => Inversion decays toward chance without shortcuts: the AUC~=0 "
              "magnitude was shortcut-driven; residual signal is weak. "
              "Soften R2's 'inverts' claim to 'fails to transfer'.")

    elapsed = time.time() - t0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "cross_generator_leakcheck_R2b",
        "purpose": "Test whether the R2 cross-generator AUC inversion survives "
                   "removal of synthetic_test shortcut features.",
        "shortcut_thresholds": {"synth_min": SHORTCUT_HI, "cifake_max": WEAK_LO},
        "per_feature_separability": {
            d: {FEATURE_NAMES[j]: round(float(sep[d][j]), 4)
                for j in range(29)} for d in DATASETS
        },
        "shortcut_features": [FEATURE_NAMES[j] for j in shortcut_idx],
        "n_features_all": 29,
        "n_features_clean": len(keep_idx),
        "matrix_all_features": m_all,
        "matrix_shortcut_removed": m_clean,
        "mean_cross_gen_auc_all": round(mean_all, 4),
        "mean_cross_gen_auc_clean": round(mean_clean, 4),
        "verdict_all": verdict_all,
        "verdict_clean": verdict_clean,
        "inversion_survives_leak_removal": bool(inversion_survives),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": stamp,
        "label_convention": "fake=1, real=0",
    }
    out_path = DATA / f"cross_generator_leakcheck_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

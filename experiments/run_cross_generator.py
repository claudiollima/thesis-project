#!/usr/bin/env python3
"""
Cross-generator generalization experiment — no torch required.

The central empirical claim of the thesis is that content-based detectors
trained on one generator fail to generalize to another (Ren et al. 2602.07814
"no universal detector"; Pirogov et al. 2507.21905 "in the wild" collapse).
So far I only had an *in-domain* number on CIFAKE (Aug 31). This experiment
puts a real number on the *generalization gap* using my own pipeline.

Design — a 2x2 train/test matrix over two independently-sourced, balanced
image sets that use DIFFERENT generators:
    - CIFAKE          (train 800/800, val 200/200)
    - synthetic_test  (train 400/400, val 100/100)

    train \\ test  | CIFAKE-val | synthetic-val
    --------------+------------+--------------
    CIFAKE        | in-domain  | CROSS-GEN
    synthetic     | CROSS-GEN  | in-domain

If content forensic features generalized, the off-diagonal (cross-generator)
cells would match the diagonal (in-domain) cells. My thesis predicts a large
drop off-diagonal. This script measures exactly that drop.

Features are IDENTICAL to run_cifake_lightweight.py (29 forensic features:
color stats + radial FFT spectrum + noise residual) so the two experiments are
directly comparable. Label convention: fake = 1 (positive), real = 0.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, confusion_matrix,
)

DATA = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA


# ---------------------------------------------------------------------------
# Feature extraction (verbatim from run_cifake_lightweight.py for comparability)
# ---------------------------------------------------------------------------
def radial_spectrum(gray: np.ndarray, n_bins: int = 16) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    bins = np.linspace(0, r.max(), n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if m.any():
            out[i] = mag[m].mean()
    return out


def color_stats(arr: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(3):
        ch = arr[:, :, c].astype(np.float64) / 255.0
        mean = ch.mean()
        std = ch.std() + 1e-8
        skew = (((ch - mean) / std) ** 3).mean()
        feats.extend([mean, std, skew])
    return np.array(feats, dtype=np.float64)


def noise_residual_stats(img: Image.Image, gray: np.ndarray) -> np.ndarray:
    blur = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(1.0)),
                      dtype=np.float64)
    resid = gray - blur
    return np.array([
        resid.mean(), resid.std(),
        np.abs(resid).mean(), np.percentile(np.abs(resid), 90),
    ], dtype=np.float64)


def extract_features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    return np.concatenate([
        color_stats(arr),
        radial_spectrum(gray, n_bins=16),
        noise_residual_stats(img, gray),
    ])


def load_split(dataset: str, split: str):
    X, y = [], []
    for label_name, label in (("real", 0), ("fake", 1)):
        d = DATA / dataset / split / label_name
        for p in sorted(d.glob("*.jpg")):
            X.append(extract_features(p))
            y.append(label)
    return np.vstack(X), np.array(y)


def evaluate(model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    t0 = time.time()
    datasets = ["cifake", "synthetic_test"]

    print("Loading + featurizing all splits...")
    train = {d: load_split(d, "train") for d in datasets}
    val = {d: load_split(d, "val") for d in datasets}
    for d in datasets:
        Xtr, ytr = train[d]
        Xva, yva = val[d]
        print(f"  {d}: train {Xtr.shape[0]} ({int((ytr==0).sum())}r/"
              f"{int((ytr==1).sum())}f), val {Xva.shape[0]} "
              f"({int((yva==0).sum())}r/{int((yva==1).sum())}f)")

    # Fit one logistic-regression detector per training set, evaluate on BOTH
    # validation sets (in-domain diagonal + cross-generator off-diagonal).
    matrix = {}
    for tr in datasets:
        Xtr, ytr = train[tr]
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
        model.fit(Xtr, ytr)
        matrix[tr] = {}
        for te in datasets:
            Xva, yva = val[te]
            res = evaluate(model, Xva, yva)
            res["kind"] = "in_domain" if tr == te else "cross_generator"
            matrix[tr][te] = res
            tag = "IN-DOMAIN " if tr == te else "CROSS-GEN "
            print(f"\n[{tag}] train={tr:14s} test={te}")
            print(f"    F1 {res['f1']:.4f} | AUC {res['auc']:.4f} | "
                  f"Acc {res['accuracy']:.4f} | "
                  f"CM[[TN,FP],[FN,TP]] {res['confusion_matrix']}")

    # Generalization gap = in-domain F1 minus cross-generator F1, per train set.
    gaps = {}
    for tr in datasets:
        others = [te for te in datasets if te != tr]
        indom = matrix[tr][tr]["f1"]
        cross = np.mean([matrix[tr][te]["f1"] for te in others])
        gaps[tr] = {
            "in_domain_f1": round(indom, 4),
            "mean_cross_gen_f1": round(float(cross), 4),
            "gap": round(float(indom - cross), 4),
        }
    mean_gap = float(np.mean([g["gap"] for g in gaps.values()]))

    print("\n=== Generalization gap (in-domain F1 - cross-generator F1) ===")
    for tr, g in gaps.items():
        print(f"  train={tr:14s} in-domain {g['in_domain_f1']:.3f} -> "
              f"cross {g['mean_cross_gen_f1']:.3f}  gap {g['gap']:+.3f}")
    print(f"  MEAN GAP: {mean_gap:+.3f}")

    elapsed = time.time() - t0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "cross_generator_generalization",
        "datasets": {
            "cifake": "CIFAKE (dragonintelligence, 224x224)",
            "synthetic_test": "independent synthetic set, 224x224",
        },
        "classifier": "LogisticRegression on 29 forensic features "
                      "(color9 + radial_fft16 + noise_residual4)",
        "label_convention": "fake=1, real=0",
        "matrix": matrix,
        "generalization_gap": gaps,
        "mean_gap": round(mean_gap, 4),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": stamp,
        "note": "torch unavailable; classical forensic-feature baseline. "
                "Provenance of synthetic_test unverified beyond directory "
                "labels; treat as a second-generator proxy.",
    }
    out_path = OUT_DIR / f"cross_generator_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

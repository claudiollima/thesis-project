#!/usr/bin/env python3
"""
R3 — three-generator cross-generalization matrix (no torch required).

R2/R2b established the cross-generator collapse on TWO sets (CIFAKE and a
weak, trivially-separable `synthetic_test` proxy). Dr. Santos' gate for R3:
a genuinely THIRD, provenance-known generator, to confirm the inversion is not
a two-set fluke.

Third set added here: **DeepFakeFace / text2img** (arXiv:2309.02218).
    - real: WIKI face photos (`wiki.zip`)
    - fake: Stable-Diffusion text2img regenerations of the SAME identities
      (`text2img.zip`) — paired real/fake, same content source, differing ONLY
      by generator. This is exactly the "paired real-fake from same source"
      design the GenD notes recommend to avoid content shortcut learning.
    - subset fetched via HTTP range requests from the zips; train identities
      (250) are DISJOINT from val identities (75) — no identity leakage.

Design — a 3x3 train/test matrix. Diagonal = in-domain, off-diagonal =
cross-generator. Thesis prediction: the off-diagonal collapses.

    train \\ test | cifake | synthetic_test | deepfakeface_t2i
    -------------+--------+----------------+-----------------
    cifake       | in-dom | cross          | cross
    synthetic    | cross  | in-dom         | cross
    deepfake     | cross  | cross          | in-dom

Confound control: all fakes in DeepFakeFace are 512x512 (SD output) while its
reals are native WIKI sizes. To stop raw image dimensions from acting as a
shortcut — and to make all three sets directly comparable — EVERY image is
resized to 224x224 before feature extraction (identity for cifake/synthetic,
which are already 224). The 29 forensic features are otherwise IDENTICAL to
R1/R2 (imported verbatim), so numbers are comparable across R1-R3.

Label convention: fake = 1 (positive), real = 0.
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

# Reuse the EXACT feature primitives from R2 so R3 is directly comparable.
from run_cross_generator import (
    color_stats, radial_spectrum, noise_residual_stats,
)

DATA = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA
IMG_SIZE = 224  # common resolution -> removes raw-dimension shortcut


def extract_features(path: Path) -> np.ndarray:
    """Identical 29 features as R1/R2, but on a resolution-normalized image."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE),
                                                 Image.BILINEAR)
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
    datasets = ["cifake", "synthetic_test", "deepfakeface_t2i"]

    print("Loading + featurizing all splits (resize->224, 29 features)...")
    train = {d: load_split(d, "train") for d in datasets}
    val = {d: load_split(d, "val") for d in datasets}
    for d in datasets:
        Xtr, ytr = train[d]
        Xva, yva = val[d]
        print(f"  {d:16s}: train {Xtr.shape[0]} ({int((ytr==0).sum())}r/"
              f"{int((ytr==1).sum())}f), val {Xva.shape[0]} "
              f"({int((yva==0).sum())}r/{int((yva==1).sum())}f)")

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
            print(f"  [{tag}] train={tr:16s} test={te:16s} "
                  f"F1 {res['f1']:.4f} | AUC {res['auc']:.4f} | "
                  f"Acc {res['accuracy']:.4f}")

    # Per-train-set generalization gap.
    gaps = {}
    for tr in datasets:
        others = [te for te in datasets if te != tr]
        indom = matrix[tr][tr]["f1"]
        cross = float(np.mean([matrix[tr][te]["f1"] for te in others]))
        gaps[tr] = {
            "in_domain_f1": round(indom, 4),
            "mean_cross_gen_f1": round(cross, 4),
            "gap": round(indom - cross, 4),
        }
    mean_gap = float(np.mean([g["gap"] for g in gaps.values()]))

    # Off-diagonal AUC summary (the polarity-inversion signal).
    cross_aucs = {f"{tr}->{te}": matrix[tr][te]["auc"]
                  for tr in datasets for te in datasets if tr != te}
    mean_cross_auc = float(np.mean(list(cross_aucs.values())))
    inverted = {k: v for k, v in cross_aucs.items() if v < 0.5}

    print("\n=== Generalization gap (in-domain F1 - mean cross-gen F1) ===")
    for tr, g in gaps.items():
        print(f"  train={tr:16s} in-domain {g['in_domain_f1']:.3f} -> "
              f"cross {g['mean_cross_gen_f1']:.3f}  gap {g['gap']:+.3f}")
    print(f"  MEAN GAP: {mean_gap:+.3f}")
    print(f"\n  mean cross-generator AUC: {mean_cross_auc:.3f}  "
          f"({len(inverted)}/{len(cross_aucs)} cells inverted, AUC<0.5)")

    elapsed = time.time() - t0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "cross_generator_3way",
        "datasets": {
            "cifake": "CIFAKE (dragonintelligence, 224x224)",
            "synthetic_test": "independent synthetic proxy, 224x224 (weak, "
                              "trivially separable - see R2b)",
            "deepfakeface_t2i": "DeepFakeFace text2img (arXiv:2309.02218): "
                                "WIKI real faces vs SD text2img fakes of same "
                                "identities; train ids disjoint from val ids",
        },
        "img_size": IMG_SIZE,
        "classifier": "LogisticRegression on 29 forensic features "
                      "(color9 + radial_fft16 + noise_residual4), resize->224",
        "label_convention": "fake=1, real=0",
        "matrix": matrix,
        "generalization_gap": gaps,
        "mean_gap": round(mean_gap, 4),
        "cross_generator_aucs": {k: round(v, 4) for k, v in cross_aucs.items()},
        "mean_cross_generator_auc": round(mean_cross_auc, 4),
        "inverted_cells": {k: round(v, 4) for k, v in inverted.items()},
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": stamp,
        "note": "torch unavailable; classical forensic-feature baseline. "
                "All images resized to 224 to remove raw-dimension shortcut. "
                "DeepFakeFace subset fetched via HTTP range from the source "
                "zips; provenance is known (SD text2img over WIKI faces).",
    }
    out_path = OUT_DIR / f"cross_generator_3way_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

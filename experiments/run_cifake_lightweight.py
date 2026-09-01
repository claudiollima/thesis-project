#!/usr/bin/env python3
"""
Lightweight CIFAKE deepfake detector — no torch required.

Motivation (Prof. Santos, week of Aug 31): turn the machine on and produce ONE
real result. torch is not installed in this environment, so instead of a deep
CNN we extract hand-crafted, forensics-inspired features and train classical
classifiers. This still yields a genuine end-to-end F1/AUC on the CIFAKE val
split that we can stand behind.

Feature families (all cheap, all computed from the raw RGB pixels):
  1. Color statistics      — per-channel mean/std/skew (diffusion images often
                             have subtly different color distributions).
  2. Frequency / spectral  — radial average of the 2D FFT magnitude. GAN/diffusion
                             images leave characteristic spectral fingerprints
                             (Zhang et al. 2019; Wang et al. CNN-generated 2020).
  3. Noise residual        — high-pass (image minus 3x3 blur) energy stats. Real
                             camera noise vs. synthesis noise differ (SRM-style).

Train: data/cifake/train  (800 real / 800 fake)
Val:   data/cifake/val    (200 real / 200 fake)

Label convention: fake = 1 (positive class), real = 0.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, confusion_matrix,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "cifake"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------
def radial_spectrum(gray: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Radially-averaged log-magnitude of the 2D FFT."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = r.max()
    bins = np.linspace(0, r_max, n_bins + 1)
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if m.any():
            out[i] = mag[m].mean()
    return out


def color_stats(arr: np.ndarray) -> np.ndarray:
    """Per-channel mean, std, and skew for an HxWx3 uint8 array."""
    feats = []
    for c in range(3):
        ch = arr[:, :, c].astype(np.float64) / 255.0
        mean = ch.mean()
        std = ch.std() + 1e-8
        skew = (((ch - mean) / std) ** 3).mean()
        feats.extend([mean, std, skew])
    return np.array(feats, dtype=np.float64)


def noise_residual_stats(img: Image.Image, gray: np.ndarray) -> np.ndarray:
    """High-pass residual energy statistics."""
    blur = np.asarray(img.convert("L").filter(ImageFilter.GaussianBlur(1.0)),
                      dtype=np.float64)
    resid = gray - blur
    return np.array([
        resid.mean(),
        resid.std(),
        np.abs(resid).mean(),
        np.percentile(np.abs(resid), 90),
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


def load_split(split: str):
    X, y, paths = [], [], []
    for label_name, label in (("real", 0), ("fake", 1)):
        d = DATA_ROOT / split / label_name
        files = sorted(d.glob("*.jpg"))
        for p in files:
            X.append(extract_features(p))
            y.append(label)
            paths.append(str(p))
    return np.vstack(X), np.array(y), paths


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading + featurizing train split...")
    Xtr, ytr, _ = load_split("train")
    print(f"  train: {Xtr.shape[0]} samples, {Xtr.shape[1]} features "
          f"({int((ytr==0).sum())} real / {int((ytr==1).sum())} fake)")

    print("Loading + featurizing val split...")
    Xva, yva, _ = load_split("val")
    print(f"  val:   {Xva.shape[0]} samples "
          f"({int((yva==0).sum())} real / {int((yva==1).sum())} fake)")

    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xva)[:, 1]
        pred = (prob >= 0.5).astype(int)
        res = {
            "f1": float(f1_score(yva, pred)),
            "auc": float(roc_auc_score(yva, prob)),
            "accuracy": float(accuracy_score(yva, pred)),
            "confusion_matrix": confusion_matrix(yva, pred).tolist(),
        }
        results[name] = res
        print(f"\n[{name}]")
        print(f"  F1       : {res['f1']:.4f}")
        print(f"  AUC      : {res['auc']:.4f}")
        print(f"  Accuracy : {res['accuracy']:.4f}")
        print(f"  Confusion [[TN,FP],[FN,TP]]: {res['confusion_matrix']}")

    elapsed = time.time() - t0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "dataset": "CIFAKE (dragonintelligence, 224x224 upscaled)",
        "train_size": int(Xtr.shape[0]),
        "val_size": int(Xva.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "feature_families": ["color_stats(9)", "radial_spectrum(16)",
                             "noise_residual(4)"],
        "label_convention": "fake=1, real=0",
        "models": results,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": stamp,
        "note": "torch unavailable; classical forensic-feature baseline.",
    }
    out_path = OUT_DIR / f"cifake_lightweight_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results -> {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

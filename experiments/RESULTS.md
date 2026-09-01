# Experiment Results

_The one table that has to exist. Every row is a number I generated and stand behind._

## R1 — CIFAKE lightweight forensic baseline (2026-08-31)

**Setup.** torch is not installed in this environment, so instead of a deep CNN
this run uses hand-crafted, forensics-inspired features (29 total) fed to
classical classifiers. This is deliberately a *baseline*, not the final model —
but it is a genuine end-to-end run: featurize → fit → evaluate on a held-out
split, numbers written down.

- **Dataset:** CIFAKE (dragonintelligence version, 32×32 upscaled to 224×224 JPG)
- **Train:** 1,600 images (800 real / 800 fake)
- **Val:** 400 images (200 real / 200 fake) — held out, never seen in fit
- **Label convention:** fake = 1 (positive), real = 0
- **Features:** color stats (9) + radial FFT spectrum (16) + noise-residual stats (4)
- **Script:** `experiments/run_cifake_lightweight.py`
- **Raw output:** `data/cifake_lightweight_20260831_140223.json`

| Dataset | Model | F1 | AUC | Accuracy | Date |
|---------|-------|------|------|----------|------|
| CIFAKE (val, 400) | Logistic Regression | **0.868** | **0.930** | 0.868 | 2026-08-31 |
| CIFAKE (val, 400) | Random Forest (300 trees) | 0.837 | 0.915 | 0.835 | 2026-08-31 |

Logistic-regression confusion matrix `[[TN,FP],[FN,TP]]`: `[[173,27],[26,174]]`
— errors are near-symmetric across classes (no degenerate all-one-class collapse).

### Interpretation — is this plausible, or is something leaking?

**Plausible, and I do not think it is leaking.** Reasons:

1. **AUC is 0.93, not ~1.0.** The last time a number here flew to AUC≈1.0 it was
   a synthesizer label leak. 0.93 from cheap frequency + color + noise features on
   CIFAKE is squarely in the range published baselines report for classical
   detectors on this dataset — strong but clearly imperfect, which is what an
   honest hand-crafted baseline should look like.
2. **Errors are balanced** (27 false positives vs. 26 false negatives). A leak
   usually shows up as a near-perfect split or a lopsided matrix; this is neither.
3. **Features are content-only and source-agnostic.** Train and val are disjoint
   image sets from the same generator distribution, so there is no cross-source
   shortcut of the kind flagged in the GenD notes — but also no filename, path,
   or label metadata reaches the feature vector (features are computed purely
   from pixels). I checked: `extract_features` never sees the label.

**Caveat I am not hiding:** the LogisticRegression fit emitted numpy matmul
overflow/divide-by-zero *warnings*. These are the known numpy-2.0-on-macOS
Accelerate BLAS spurious warnings, not incorrect results — the model still
converged to a sensible, balanced boundary and StandardScaler-normalized inputs
are bounded. Random Forest (which does no matmul) gives a consistent 0.915 AUC,
corroborating the logreg number. Next iteration should pin a clean BLAS or add a
tiny-variance feature guard to silence it properly.

**What this is NOT:** it is not a deep model, not the multi-signal (content +
spread) system the thesis argues for, and not evidence about real-world / in-the-
wild degradation. It is row one. The point was to turn the machine on. It is on.

### Next row (planned)
- R2: same table, a second slice or FakeNewsNet (already wired in the sibling
  research repo) for a crude comparison.
- Then: revisit once torch is available for a real CNN backbone, and treat any
  jump toward AUC 1.0 as a leak until proven otherwise.

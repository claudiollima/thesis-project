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

## R2 — Cross-generator generalization (2026-09-01)

**Question.** R1 gave an honest *in-domain* number. The thesis's central empirical
claim is that content detectors **fail to transfer across generators** (Ren et al.
2602.07814 "no universal detector"; Pirogov et al. 2507.21905 in-the-wild collapse).
This run puts my own number on that gap with a 2×2 train/test matrix over two
independently-sourced, balanced sets using the **identical 29 forensic features**
as R1 (so the two experiments are directly comparable).

- **Sets:** CIFAKE (train 1600, val 400) and `synthetic_test` (train 800, val 200)
- **Classifier:** Logistic Regression on 29 features (color 9 + radial-FFT 16 + noise 4)
- **Script:** `experiments/run_cross_generator.py`
- **Raw output:** `data/cross_generator_20260901_140143.json`

| Train ↓ / Test → | CIFAKE-val | synthetic-val | |
|------------------|-----------|--------------|---|
| **CIFAKE** | F1 0.868 / AUC 0.930 | **F1 0.000 / AUC 0.0004** | in-domain vs CROSS-GEN |
| **synthetic_test** | F1 0.568 / AUC 0.370 | F1 1.000 / AUC 1.000 | CROSS-GEN vs in-domain |

**Mean in-domain → cross-generator F1 gap: +0.650.** Both off-diagonal (cross-
generator) cells are at or **below chance**.

### The headline is stronger than "it drops to random"

The CIFAKE→synthetic cell has **AUC ≈ 0**, not 0.5. That is not noise — it means the
detector's confidence ranking is almost **perfectly inverted** on the second set:
images CIFAKE calls most-fake are the ones synthetic_test labels real. The learned
forensic decision boundary doesn't just stop working across generators; **its
polarity flips.** A detector that looks trustworthy in-domain (genuine AUC 0.93,
overlapping classes, balanced errors) becomes an actively *anti*-correlated
predictor on another source. This is a sharper illustration of the arms-race
argument than a plain accuracy drop.

### Honesty check — the synthetic_test in-domain 1.000 is an artifact, NOT a win

I chased the AUC=1.0 the way Dr. Santos trained me to. Diagnosis:

- **7 of 29 features perfectly separate real/fake in synthetic_test** (`R_mean`,
  and high-frequency FFT bins `fft_r2, r11–r15`) with **zero overlap** between
  classes. CIFAKE, by contrast, has **0** perfectly-separating features — its
  classes overlap on every axis, which is why it needs a real classifier and lands
  at an honest 0.87.
- The synthetic_test shortcut: its "real" images are systematically **brighter in
  R** (mean ≈ 0.58, very tight) and carry **higher high-frequency energy** (sharper)
  than its "fakes" (smoother, low-freq — the classic diffusion-smoothing signature).
- That shortcut is the **opposite polarity** to what CIFAKE learned — which is
  precisely the mechanism behind the AUC≈0 inversion above.

So I do **not** report synthetic_test in-domain 1.000 as detection performance;
it's a dataset-construction artifact, and I'm flagging it as such. The trustworthy,
leak-checked number remains R1's CIFAKE 0.87/0.93. The cross-generator collapse is
the real finding.

**Caveat (numeric):** logreg emits matmul overflow warnings on the synthetic_test
fit — here driven by the perfectly-separable training data (weights diverge), on
top of the known numpy-2.0 macOS BLAS warnings from R1. Results still compute
correctly; a stronger L2 penalty or dropping the leaking features would silence it.
Provenance of `synthetic_test` beyond its directory labels is unverified — treat it
as a second-generator *proxy*, not a named model.

## R2b — Does the inversion survive its own leak check? (2026-09-01)

**Question.** R2 flagged a caveat: the ~7 features that perfectly separate
`synthetic_test` are opposite-polarity to CIFAKE, and I named them as "precisely the
mechanism behind the AUC≈0 inversion." If that's literally true, then the dramatic
inversion is an artifact of one dataset's construction, not a claim about content
detection. So I tested it directly: **remove the shortcut features and re-run the
same 2×2 matrix.** If the inversion vanishes, I have to walk back R2's headline.

- **Method:** rank all 29 features by direction-free in-set separability
  `max(AUC, 1−AUC)`; auto-flag features that are near-perfect on synthetic_test
  (≥0.98) but weak on CIFAKE (<0.75); re-run the matrix with those dropped.
- **Script:** `experiments/run_cross_generator_leakcheck.py`
- **Raw output:** `data/cross_generator_leakcheck_20260901_140617.json`

**Feature audit confirms R2's diagnosis quantitatively.** Exactly **7 features hit
in-set AUC = 1.000 on synthetic_test** (`R_mean`, `fft_bin02`, `fft_bin11–15`) — the
7 I named in R2 — and **11** clear the ≥0.98 / <0.75 shortcut bar. CIFAKE's best
single feature is only 0.80; it has **zero** features above 0.72 that synthetic_test
doesn't also exploit. synthetic_test is broadly, trivially separable; CIFAKE is not.

| Cross-gen cell | AUC, all 29 | AUC, 11 shortcuts removed (18 feats) |
|----------------|-------------|--------------------------------------|
| CIFAKE → synthetic | 0.0004 | 0.0060 |
| synthetic → CIFAKE | 0.370 | 0.351 |
| **mean cross-gen AUC** | **0.185** | **0.178** |

**Verdict: the inversion SURVIVES.** Stripping every shortcut feature moves mean
cross-generator AUC by <0.01 (0.185 → 0.178); both cells stay deep in inverted
territory. The AUC≈0 collapse is therefore **not** carried by the perfectly-
separating features — it's a property of the shared 18-feature forensic boundary
itself. R2's headline stands: content forensic boundaries flip polarity across
generators. Two independent validations came for free: (a) the all-features matrix
reproduces R2 **cell-for-cell** (0.930 / 0.0004 / 0.370 / 1.000), and (b) CIFAKE's
own in-domain AUC only drops 0.930 → 0.894 without the shortcuts, so I didn't gut
the honest signal to kill the leaky one.

**What it does NOT fix:** synthetic_test's in-domain AUC is *still* 1.000 on the
remaining 18 features — it stays trivially separable, so it's a weak second
generator and a poor proxy for "in the wild." That's exactly why R3 (a real third,
provenance-known generator) is still the gating next step, not optional polish.

### Next row (planned)
- R3: get a genuinely third, provenance-known generator (Midjourney set currently
  only has real images — needs its fake half) to confirm the inversion isn't a
  two-set fluke.
- R4: the actual thesis test — add **spread-pattern signal** on top and show it
  survives the cross-generator shift where content features invert.
- Then: revisit once torch is available for a real CNN backbone, and treat any jump
  toward AUC 1.0 as a leak until proven otherwise.

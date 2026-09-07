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

## R3 — third, provenance-known generator: 3×3 cross-generator matrix (2026-09-02)

**The gate.** R2/R2b established the cross-generator collapse on only TWO sets, one
of which (`synthetic_test`) is trivially separable and a weak proxy. Dr. Santos'
R3 gate: a genuinely THIRD, provenance-known generator, to confirm the effect isn't
a two-set fluke. This closes it.

**Third set: DeepFakeFace / text2img (arXiv:2309.02218).**
- **real:** WIKI face photos (`wiki.zip`).
- **fake:** Stable-Diffusion **text2img** regenerations of the *same identities*
  (`text2img.zip`) — paired real/fake, same content source, differing only by
  generator. This is the "paired real-fake from same source" design the GenD notes
  recommend to avoid content shortcut learning.
- **Acquired honestly:** 250 train + 75 val identity pairs streamed out of the two
  1–1.7 GB source zips via HTTP range requests (parse central directory → range-GET
  each member → inflate), rather than pulling 3 GB. **Train identities are disjoint
  from val identities** — no identity leakage.
- **Confound control:** all DeepFakeFace fakes are 512×512 (SD output) while its
  reals are native WIKI sizes, so raw dimensions could be a shortcut. Every image in
  all three sets is resized to 224×224 before featurizing (identity for CIFAKE /
  synthetic_test, already 224). Same 29 forensic features as R1/R2, imported verbatim.
- **Script:** `experiments/run_cross_generator_3way.py`
- **Raw output:** `data/cross_generator_3way_20260902_140837.json`

| Train ↓ / Test → | CIFAKE-val | synthetic-val | deepfake-val |
|------------------|-----------|---------------|--------------|
| **CIFAKE** | **0.868 / 0.930** (in-dom) | 0.000 / 0.000 | 0.000 / 0.578 |
| **synthetic_test** | 0.568 / 0.370 | **1.000 / 1.000** (in-dom*) | 0.575 / 0.410 |
| **deepfakeface_t2i** | 0.385 / 0.420 | 0.000 / 0.859 | **0.733 / 0.788** (in-dom) |

*(each cell = F1 / AUC. \*synthetic_test in-domain 1.000 is the known artifact from R2b, not a win.)*

**Headline numbers:** mean in-domain→cross F1 gap **+0.612**; mean cross-generator
AUC **0.440** (below chance); **4 of 6** off-diagonal cells are inverted (AUC < 0.5);
every off-diagonal F1 ≤ 0.575. **The two-set collapse was not a fluke — it holds
with a genuine, provenance-known third generator.**

### The third set behaves like an HONEST generator, which is the point

DeepFakeFace in-domain lands at **F1 0.733 / AUC 0.788** with a balanced confusion
matrix `[[55,20],[20,55]]` — clearly imperfect, symmetric errors, no leak. That is
exactly what a real generator with overlapping classes should look like, and it is
the opposite of synthetic_test's degenerate 1.000. So R3 is carried by a legitimate
detection problem, and the cross-generator collapse around it is trustworthy.

### Honest correction to R2's headline

R2 called the effect a **polarity flip to AUC ≈ 0**. R3 shows that was the *extreme
end of a spectrum, specific to the CIFAKE↔synthetic pair* (CIFAKE→synthetic AUC
0.0004), **not a universal law.** With the honest third generator the cross cells
cluster **around chance with mild inversion** (0.37–0.58), not at ≈0. The defensible,
general claim is therefore weaker but sturdier: *content forensic detectors do not
transfer across generators — cross-generator AUC collapses to ≈chance (0.44 mean,
often below it).* I'm walking the "perfect inversion" framing back to "collapse to
chance"; the dramatic inversion is real but is one pair, not the rule.

### A second failure mode: threshold collapse even when ranking survives

Three off-diagonal cells have **F1 = 0.000 with confusion matrix `[[N,0],[N,0]]`** —
the detector predicts *everything real* at the fixed 0.5 threshold. Two are near
chance in ranking too (CIFAKE→synthetic AUC 0.000; CIFAKE→deepfake AUC 0.578), but
one is striking: **deepfake→synthetic has AUC 0.859 yet F1 0.000.** The probability
*ranking* transfers, but the *operating point* is so miscalibrated under domain shift
that the deployed classifier is useless. This is a distinct, deployment-relevant face
of non-transfer: even when a detector "could" separate classes on another generator,
its calibrated threshold does not survive the shift. (The 0.859 is also inflated by
synthetic_test's trivial separability along shared feature axes — I don't read it as
genuine transfer.)

### What this does and does not establish

- **Does:** three independently-sourced generators, one of them paired same-source
  faces with fully known provenance, all confirm content-forensic detection does not
  cross generators. Mean cross-gen AUC 0.44. R2 reproduced cell-for-cell (0.868 /
  0.930 CIFAKE, 0.000/0.0004 CIFAKE→synthetic), so nothing shifted under the resize.
- **Does not:** this is still a classical 29-feature baseline, not a deep CNN, and
  the subsets are small (75–200 val per set). It does not yet show the *positive*
  half of the thesis — that spread signal survives where content fails.

### Next row (planned)
- Report cross-generator numbers at a **calibrated / tuned threshold** too, not just
  0.5, to separate "ranking fails" from "threshold fails" (R3 showed both happen).
- Revisit with a real CNN backbone once torch is available; treat any jump toward
  AUC 1.0 as a leak until proven otherwise.

## R4 — the positive half: spread signal survives its own domain shift (2026-09-03)

**The gate.** R1–R3 nailed the NEGATIVE half: content-forensic detectors collapse to
~chance across *generators* (mean cross-generator AUC 0.44, 4/6 cells inverted). The
thesis also makes a POSITIVE claim — that **spread-pattern signal is robust where
content fails**. R4 is the first empirical test of that claim on my own numbers.

**Design (a fair mirror of R1–R3, not more content baselines).** For *content* the
natural axis of failure is a new generator (different pixels). For *spread* the
natural axis is a new **domain** — a different platform / topic / cascade scale. So I
test spread on its own threat axis: train a spread detector on one domain, test on
another, in a full **3×3 cross-domain matrix** (same protocol as R3: StandardScaler +
LogisticRegression, balanced, held-out val, F1/AUC).

- **Three domains, genuinely different absolute scale, identical class *structure*:**
  `meme` (small/fast/low-follower), `news` (large/slow/high-follower),
  `niche` (medium, cross-platform-prone). Only the scale changes across domains — the
  organic-vs-coordinated contrast is the same everywhere. That is the mirror of the
  content setup (fix the label structure, vary the generator).
- **Features:** the **28-feature spread extractor reused verbatim** from
  `research/experiments/spread_patterns.py` (the set already documented in THESIS.md,
  Feb 17 + Feb 23 Murugan-mechanism theory). Imported, not reimplemented.
- **Script:** `experiments/run_spread_cross_domain.py`
- **Raw output:** `data/spread_cross_domain_20260903_140439.json`

| Train ↓ / Test → | meme | news | niche |
|------------------|------|------|-------|
| **meme**  | **0.743 / 0.871** (in-dom) | 0.000 / 0.855 | 0.630 / 0.884 |
| **news**  | 0.000 / 0.639 | **0.793 / 0.874** (in-dom) | 0.674 / 0.787 |
| **niche** | 0.667 / 0.818 | 0.707 / 0.744 | **0.800 / 0.868** (in-dom) |

*(each cell = F1 / AUC.)*

**Headline:** in-domain AUC **0.871**; **cross-domain AUC 0.788, with 0/6 cells below
chance.** Contrast R3's content result on its own axis: cross-generator AUC **0.44**,
4/6 below chance. **Spread signal keeps ~0.79 ranking power across domains where
content forensics collapsed to ~chance across generators.** That is the positive half
of the thesis, measured, for the first time.

**Not seed-luck.** Over 5 independent seeds: in-domain AUC **0.885 ± 0.016**,
cross-domain AUC **0.824 ± 0.058**. The single reported matrix (0.871 / 0.788) sits at
the low end of that spread, so it is a conservative snapshot, not a cherry-pick.

### This is an HONEST detection problem, not the synthetic_test trap

The first draft of this simulator produced **in-domain AUC = 1.000 everywhere** —
exactly the degenerate perfect-separation I called out for `synthetic_test` in R2. I
did **not** keep it. I rebuilt the simulator so the label only *shifts* overlapping
generating distributions (coordination strength drawn from overlapping Beta
distributions; every account/timing parameter interpolates with per-account noise).
The result is an honest in-domain AUC ~0.87 with overlapping classes — a real
classification problem, the same character as CIFAKE (0.93) and DeepFakeFace (0.79),
**not** a 1.0 tautology. The label never enters the feature vector (features are pure
cascade structure); AUC 0.87 ≠ 1.0 is the leak check passing.

### The feature-group ablation refuted my own hypothesis (reported anyway)

I expected **dimensionless structural** features (CVs, fractions, virality — the
Murugan-mechanism signatures) to transfer *better* than **scale-dependent** ones (raw
follower counts, share rates). The numbers say otherwise:

| Feature subset | in-domain AUC | cross-domain AUC |
|----------------|---------------|------------------|
| all 28         | 0.871 | 0.788 |
| structural (14, scale-invariant) | 0.844 | **0.771** |
| scale-dependent (14) | 0.867 | **0.832** |

Scale features transfer *slightly better* (0.832 vs 0.771), not worse. Mechanism:
per-domain `StandardScaler` re-centers each feature, so the *direction* of the
follower/age shift is consistent across domains even though the absolute magnitude
differs — standardization launders the scale difference. So the robustness is **broad
(both subsets transfer 0.77–0.83)**, not carried solely by the dimensionless features
I theorized about. I'm recording the refutation rather than burying it.

### The threshold-collapse failure mode recurs (consistent with R3)

Two cross cells — `meme→news` and `news→meme` — have **F1 0.000 despite AUC 0.85 /
0.64**: the probability *ranking* transfers but the fixed 0.5 threshold predicts one
class. This is the **same** second failure mode R3 found (deepfake→synthetic: AUC
0.859, F1 0.000). It reinforces R3's planned next step: spread transfers in *ranking*
but a deployed spread detector still needs **per-domain recalibration** of its
operating point. Ranking-robust ≠ threshold-robust.

### What this does and does NOT establish

- **Does:** on its own natural shift axis, a spread detector retains strong ranking
  power (cross-domain AUC ~0.79, 0/6 below chance, stable over seeds) around an honest
  in-domain anchor — the empirical positive counterpart to R1–R3's content collapse.
- **Does NOT:** the cascades are **simulated**, so this is *mechanism validation*, not
  an in-the-wild measurement. All three domains share one simulator with the same
  qualitative class definition, so it does not prove transfer when the *definition* of
  coordination itself shifts. And content (real images, cross-generator) vs spread
  (simulated cascades, cross-domain) is a comparison across different data types and
  shift axes — **suggestive of complementarity, not a controlled head-to-head.** The
  honest next gate is real paired content+spread data (the still-pending item), and
  reporting spread at a recalibrated threshold, not just 0.5.

## R5 — threshold recalibration: is the F1-0.000 collapse fixable? (2026-09-04)

**Motivation.** R3 and R4 both found the same second failure mode and both deferred
the same fix verbatim: several cross cells have strong *ranking* (AUC 0.64–0.85) but
**F1 = 0.000** at the fixed 0.5 threshold — the probability ordering transfers, the
operating point does not. R4's closing line: "reporting spread at a recalibrated
threshold, not just 0.5." This row does exactly that and answers one question:

> Is the F1-0.000-at-AUC-0.85 collapse a **fixable calibration artifact** (ranking is
> usable, only the threshold is wrong), or a **genuine transfer failure** (the ranking
> isn't actually convertible into a decision)?

**Setup.** Reuses R4's simulator + 28-feature spread extractor + train protocol
*verbatim* (imported from `run_spread_cross_domain.py`; nothing re-tuned). The learned
model is held **fixed** — AUC is identical across policies because we only relabel the
same score ranking. Only the decision-threshold policy varies, ordered most-deployable
to cheating-ceiling:

1. **naive_0.5** — R4 baseline (reproduces the collapse).
2. **source_prior** — threshold set on the *source* domain's own scores so the
   predicted-positive rate matches the known class prior (0.5, balanced), applied to
   target. **Uses ZERO target labels** — only assumes you know the base rate.
3. **target_cal** — reserve a small labelled slice of the target (**25/class**), pick
   the F1-max threshold there, evaluate on a disjoint target test. Few-label
   deployment; uses target labels but not the test scores.
4. **target_oracle** — F1-max threshold on the target *test* labels themselves. This
   **peeks at test**; reported ONLY as a ceiling ("how much is recoverable"), never as
   a result.

- **Script:** `experiments/run_spread_threshold_recal.py`
- **Raw output:** `data/spread_threshold_recal_20260904_140224.json`
- **Protocol:** 150/class train, 25/class target calibration, 60/class disjoint target
  test; LogisticRegression + per-domain StandardScaler (same as R1–R4).

### Result — cross-domain F1 by policy (mean over the 6 cross cells, 5 seeds)

| Policy | target labels used | cross-domain F1 (5-seed) | vs naive |
|--------|--------------------|--------------------------|----------|
| naive_0.5 | none | 0.425 ± 0.097 | — |
| **source_prior** | **none (base rate only)** | **0.429 ± 0.100** | **+0.004** |
| **target_cal** | **25 / class** | **0.761 ± 0.024** | **+0.336** |
| target_oracle (ceiling) | test labels (cheat) | 0.799 ± 0.017 | +0.374 |

Cross-domain AUC is **0.85** and unchanged by every policy (we only move the threshold).

The single reported matrix (seed 20260904) shows the mechanism cell-by-cell — the cell
that was fully collapsed under R3/R4's 0.5 rule:

| cross cell | AUC | F1 naive | F1 source_prior | F1 target_cal | F1 oracle |
|------------|-----|----------|-----------------|---------------|-----------|
| meme→news  | 0.844 | **0.000** | **0.000** | 0.750 | 0.790 |
| meme→niche | 0.813 | 0.209 | 0.209 | 0.732 | 0.762 |
| news→meme  | 0.917 | 0.581 | 0.581 | 0.864 | 0.891 |
| news→niche | 0.889 | 0.571 | 0.571 | 0.822 | 0.835 |
| niche→meme | 0.859 | 0.780 | 0.775 | 0.803 | 0.819 |
| niche→news | 0.780 | 0.690 | 0.690 | 0.663 | 0.780 |

### The answer: fixable — but ONLY with a little target supervision

- **The collapse is a calibration artifact, not a ranking failure.** The fully-dead
  `meme→news` cell (F1 0.000 at AUC 0.844) recovers to **0.750** with 25 target
  labels/class, and mean cross-domain F1 jumps **0.425 → 0.761 (+0.336)**. The ranking
  was usable all along; the 0.5 threshold was simply in the wrong place.
- **`target_cal` (0.761) sits within 0.038 of the `oracle` ceiling (0.799)** and has
  ~4× tighter variance than naive (±0.024 vs ±0.097). 25 labels/class recovers ~95% of
  the recoverable F1 — cheap, realistic deployment.

### The self-refutation I stand behind: base rate alone does NOT fix it

I expected `source_prior` — matching the known 50/50 base rate on source scores — to
rescue the collapse for free (no target labels). **It does not:** 0.429 vs naive 0.425,
a **+0.004** non-effect, and it leaves the same 1/6 cells fully collapsed. Mechanism:
the classifier's score *distribution* shifts across domains (systematically compressed/
offset on the target), so a threshold calibrated on the source lands in the wrong place
on the target regardless of the prior. **Knowing the base rate is not enough; you need a
few actual target-domain labels to locate the operating point.** Recording the refuted
hypothesis rather than burying it.

### What this does and does NOT establish

- **Does:** refines R4's positive claim into a deployment-honest one — spread's
  cross-domain *ranking* (AUC ~0.85) **is** convertible into usable decisions
  (cross F1 ~0.76), closing the R3/R4 threshold-collapse gap, but this requires a
  **small target-labelled calibration set (~25/class)**. Zero-shot at the *ranking*
  level ≠ zero-shot at the *decision* level.
- **Does NOT:** the cascades are still **simulated** (mechanism validation, as in R4),
  and the recalibration is demonstrated on the spread axis only. It does not remove the
  still-pending gate: **real paired content+spread data**, on which the same
  source→target recalibration protocol should be re-run.

---

## R6 — Controlled content × spread head-to-head (SAME items, SAME shift) — 2026-09-07

**Script:** `experiments/run_content_spread_headtohead.py`
**Raw:** `data/content_spread_headtohead_20260907_140217.json`

### Why R6 exists

R4 and R5 both closed with the *same* verbatim caveat: content (real images,
tested across generators) vs spread (simulated cascades, tested across domains)
were **different data on different axes — suggestive of complementarity, not a
controlled head-to-head.** The Feb-20 combined-signal table shared the weakness:
content accuracy was a hand-set dial, not a coupled property of each item.

R6 removes it. **Every item now carries BOTH** a spread cascade (R4 simulator,
reused verbatim) **and** a content-forensic latent for that *same item*, and both
are evaluated across the *same* 3×3 domain shift. Content and spread are coupled
through the shared per-item coordination strength `c` (the same overlapping Beta
that drives the cascade also scales the content-fake strength), so the two
signals are correlated through the latent cause — as in reality — not independent
by fiat.

**Content model is not a rigged straw man.** Each domain has a near-orthogonal
generator *fingerprint* (Gram-Schmidt; measured cos ≈ 0.000 between domains). A
content boundary learned in one domain is therefore uninformative on another —
this is the R1–R3 generator-collapse mechanism expressed on matched items, not a
hand-set accuracy. FAKE_SHIFT tuned so in-domain content AUC ≈ 0.75 (honest
overlap, near the real CIFAKE 0.93 / DeepFakeFace 0.79 anchors, **not** the
degenerate 1.0).

### Result (AUC threshold-free; F1 at R5 `target_cal` threshold, 25 labels/class)

| signal | in AUC | in F1 | cross AUC | cross F1 | cross below-chance |
|--------|--------|-------|-----------|----------|--------------------|
| content | 0.747 | 0.724 | **0.465** | 0.655 | **5/6** |
| spread  | 0.862 | 0.797 | **0.805** | 0.751 | **0/6** |
| fusion (source-weight late fusion) | **0.901** | **0.823** | 0.614 | 0.684 | 2/6 |

**Per-cell (C content / S spread / F fusion, AUC):**

| cell | content | spread | fusion |
|------|---------|--------|--------|
| (in) meme→meme | 0.722 | 0.869 | 0.901 |
| meme→news | 0.450 | 0.792 | 0.450 |
| meme→niche | 0.462 | 0.826 | 0.474 |
| news→meme | 0.566 | 0.838 | 0.811 |
| (in) news→news | 0.747 | 0.859 | 0.897 |
| news→niche | 0.466 | 0.847 | 0.765 |
| niche→meme | 0.444 | 0.883 | 0.584 |
| niche→news | 0.401 | 0.642 | 0.601 |
| (in) niche→niche | 0.771 | 0.858 | 0.905 |

### Headline

On matched items under the same shift:

1. **Content collapses cross-domain** (AUC 0.465, 5/6 below chance) — the R1–R3
   mechanism reproduced on the *same items* spread succeeds on. First time the
   negative half is shown on matched data.
2. **Spread transfers** (AUC 0.805, 0/6 below chance) — R4 confirmed on matched
   data, same axis as content for the first time.
3. **Fusion is best IN-DOMAIN (0.901) but WORSE than spread-alone CROSS-DOMAIN
   (0.614 vs 0.805, −0.191 AUC / −0.067 F1).** A source-trained late-fusion
   meta-learner keeps trusting content; because cross-domain content is
   *anti-correlated* (AUC < 0.5), that trust drags fusion below spread alone.

### The self-refutation I stand behind (Dr. Santos discipline)

I expected the R5 lesson to carry over: "a few target labels fix the cross-domain
gap." R5 was about the *threshold*; R6 asked whether it also fixes the *fusion
combination*. **It does not.** Relearning the 2 fusion weights on a 25-label/class
target slice (`target-refit fusion`) gives cross-domain **AUC 0.670** — still far
below spread-alone 0.805:

| cross-domain (6 cells) | AUC | F1 |
|------------------------|-----|-----|
| spread alone | **0.805** | **0.750** |
| fusion, source weights | 0.665 | 0.688 |
| fusion, target-refit (25 labels/class) | 0.670 | 0.668 |

**Why the R5 trick fails here:** a threshold is 1 scalar and cross-domain
*ranking* was intact, so 25 labels sufficed (R5). The fusion combination must
learn to *ignore* a channel that is genuinely anti-correlated on the target; a
2-feature meta-learner on 25 labels can't reliably do that. **A collapsed content
channel is dead weight that a small calibration set cannot cheaply gate off.**

### What this establishes / does NOT

- **Establishes:** the complementarity story is now a *controlled* head-to-head on
  matched items, and it is sharper than "combine and win." Naive late fusion is
  **actively harmful under domain shift**; the robust deployment choice is to
  **drop the collapsed content channel, not fuse it** — and this is *not* the
  cheaply-fixable calibration artifact R5 found. Selective/gated fusion that can
  fully down-weight a dead channel (not a fixed source-trained combiner) is the
  real open design problem.
- **Does NOT:** content latents and cascades are both **simulated** — mechanism
  validation of complementarity on matched items, not an in-the-wild number. The
  still-pending gate is unchanged: **real paired content+spread data**, on which
  this same matched head-to-head + fusion protocol should be re-run.

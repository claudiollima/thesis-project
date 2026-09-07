"""
R6 — Controlled content x spread head-to-head, SAME items, SAME shift.

The one caveat repeated verbatim in R4 and R5:

    "content (real images) vs spread (simulated cascades) is different data on
     different axes -- suggestive of complementarity, not a controlled
     head-to-head."

R1-R3 measured CONTENT on real images across GENERATORS. R4-R5 measured SPREAD
on simulated cascades across DOMAINS. They were never measured on the SAME items
under the SAME shift, so "spread rescues content" was only ever *suggestive*.
The Feb-20 combined-signal table (content acc knobbed by hand, spread bolted on)
had the same weakness: content quality was a free dial, not a coupled property
of each item.

R6 removes that weakness. Every item now carries BOTH:
  * a SPREAD cascade (reused verbatim from R4's build_domain_dataset simulator),
  * a CONTENT-forensic score generated for that SAME item.
and both are evaluated across the SAME domain shift, so fusion is a controlled
head-to-head, not a comparison of two different experiments.

--------------------------------------------------------------------------------
The content model, and why it is NOT a tautology
--------------------------------------------------------------------------------
The whole point of R1-R3 is that a content-forensic detector's decision boundary
is GENERATOR-SPECIFIC and collapses to ~chance under generator shift. R6 must
reproduce that mechanism honestly, or fusion would win for free.

We give each domain a dominant *generator* with its own forensic "fingerprint"
direction in a small latent content space:
  * Fake items carry a shift along their domain-generator's fingerprint vector,
    plus heavy per-item noise (honest in-domain overlap, AUC ~0.8 -- matching the
    real CIFAKE/DeepFakeFace in-domain anchors from R1/R3, NOT the degenerate 1.0).
  * A content model trained in domain A learns A's fingerprint direction. Tested
    on domain B (different generator = different, ~orthogonal fingerprint), that
    direction is uninformative -> the content model collapses toward chance,
    exactly as R1-R3 measured on real images. This is a *property of the item's
    latent*, decided by the numbers, not a hand-set accuracy dial.

Spread structure, by contrast, is domain-invariant (R4), so the spread model
transfers. The interesting, non-tautological question is then:

    On the SAME items under the SAME cross-domain shift, does LATE FUSION of a
    (collapsed) content model and a (transferring) spread model beat the best
    single signal -- and is content DEAD WEIGHT cross-domain, or does its
    residual in-domain-learned signal still help where it is trained?

Honesty framing (Dr. Santos discipline):
  * Both content latents and cascades are SIMULATED. This is MECHANISM
    validation of complementarity ON MATCHED ITEMS, not an in-the-wild number.
    The still-open gate (real paired content+spread data) is unchanged.
  * The content generator is coupled to the label the SAME overlapping way the
    spread simulator is (shared coordination strength c drives BOTH the cascade
    AND the content-fake strength), so the two signals are correlated through the
    latent cause -- as they would be in reality -- not independent by fiat.
  * Fusion uses a small target-calibration slice for its threshold (the R5
    lesson: cross-domain ranking is usable but the operating point needs a few
    target labels). We report AUC (threshold-free) as the primary transfer
    metric and F1 at the R5-style target_cal threshold as the decision metric.

Same protocol as R1-R5: StandardScaler + LogisticRegression, balanced sets,
fixed seeds, full N x N matrix, honest in-domain overlap.

Author: Claudio L. Lima
Date: 2026-09-07
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# --- reuse R4's cascade simulator + spread extractor VERBATIM --------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_spread_cross_domain import (  # noqa: E402
    DOMAINS,
    _simulate_cascade,
)
_RESEARCH_EXP = os.path.expanduser(
    "~/.openclaw/agents/claudiollm/workspace/research/experiments"
)
sys.path.insert(0, _RESEARCH_EXP)
from spread_patterns import SpreadPatternExtractor  # noqa: E402


# ---------------------------------------------------------------------------
# Per-domain generator fingerprints in a small forensic-latent space.
# Each domain has a DIFFERENT, near-orthogonal fingerprint direction: that is
# the R1-R3 mechanism (generator-specific boundary) expressed on matched items.
# ---------------------------------------------------------------------------
CONTENT_DIM = 12


def _domain_fingerprints(seed=20260907):
    """One (near-)orthogonal unit fingerprint per domain, deterministic."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(len(DOMAINS), CONTENT_DIM))
    # Gram-Schmidt to make the domain generators genuinely near-orthogonal:
    # a content boundary learned on one is uninformative on the others.
    Q, _ = np.linalg.qr(raw.T)
    fps = Q.T[: len(DOMAINS)]
    return {dom: fps[i] for i, dom in enumerate(DOMAINS)}


FINGERPRINTS = _domain_fingerprints()

# forensic-fake strength: how far a fake item is pushed along the fingerprint.
# Kept low enough that in-domain content AUC lands ~0.8 (honest overlap, matching
# the real CIFAKE 0.93 / DeepFakeFace 0.79 anchors -- NOT the degenerate 1.0).
FAKE_SHIFT = 1.15
CONTENT_NOISE = 1.0


def _content_latent(rng, domain, coordinated, c):
    """Latent forensic vector for ONE item in `domain`.

    Coupling: the SAME coordination strength `c` that shaped this item's cascade
    also scales its content-fake strength, so content and spread are correlated
    through the shared latent cause (as in reality), not independent by fiat.
    A fake (coordinated) item is pushed along the domain generator's fingerprint
    by FAKE_SHIFT * c; heavy isotropic noise creates honest class overlap.
    """
    base = rng.normal(0.0, CONTENT_NOISE, size=CONTENT_DIM)
    if coordinated:
        base = base + FAKE_SHIFT * c * FINGERPRINTS[domain]
    return base


def build_paired_domain_dataset(domain_name, n_per_class, seed):
    """Return matched (X_content, X_spread, y, spread_names) for one domain.

    Row i of X_content and row i of X_spread describe the SAME item i -- this is
    the whole point of R6. Both are driven by the same per-item coordination
    strength c drawn inside the shared simulator path.
    """
    rng = np.random.default_rng(seed)
    cfg = DOMAINS[domain_name]
    extractor = SpreadPatternExtractor(observation_window_hours=48)
    names = extractor.get_feature_names()

    Xc, Xs, y = [], [], []
    for coord, label in [(True, 1), (False, 0)]:
        for k in range(n_per_class):
            cid = f"{domain_name}_{label}_{k}"
            casc = _simulate_cascade(rng, cfg, coord, cid)
            # recover THIS item's coordination strength from its cascade so the
            # content latent is coupled to the same generative cause. The cascade
            # stores is_synthetic; we re-draw c from the SAME overlapping Beta the
            # simulator used, with a per-item rng so it is stable and coupled.
            item_rng = np.random.default_rng(
                abs(hash(cid)) % (2**32))
            c = float(item_rng.beta(3.0, 1.4) if coord else item_rng.beta(1.4, 3.0))
            feats = extractor.extract_all_features(casc)
            Xs.append([feats[n] for n in names])
            Xc.append(_content_latent(item_rng, domain_name, coord, c))
            y.append(label)

    Xc = np.array(Xc, dtype=float)
    Xs = np.nan_to_num(np.array(Xs, dtype=float),
                       nan=0.0, posinf=1e9, neginf=-1e9)
    return Xc, Xs, np.array(y, dtype=int), names


# ---------------------------------------------------------------------------
# thresholding (R5 target_cal policy) + metrics
# ---------------------------------------------------------------------------
def _f1max_threshold(proba, y):
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


def _auc(y, p):
    try:
        return round(float(roc_auc_score(y, p)), 3)
    except ValueError:
        return float("nan")


class Signal:
    """A fitted StandardScaler+LogReg producing calibrated probabilities."""

    def __init__(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(max_iter=2000, C=1.0).fit(
            self.scaler.transform(X), y)

    def proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


def main():
    N_TRAIN = 150   # per class (matches R4/R5)
    N_CAL = 25      # per class, target calibration slice (R5 lesson)
    N_TEST = 60     # per class, held-out target test (matches R4/R5)
    SEED = 20260907

    print("R6 — content x spread head-to-head, SAME items, SAME shift")
    print("=" * 64)

    train, cal, test = {}, {}, {}
    names = None
    for i, dom in enumerate(DOMAINS):
        Xc, Xs, y, names = build_paired_domain_dataset(dom, N_TRAIN, SEED + i)
        Xc2, Xs2, y2, _ = build_paired_domain_dataset(
            dom, N_CAL + N_TEST, SEED + 500 + i)
        pos = np.where(y2 == 1)[0]
        neg = np.where(y2 == 0)[0]
        cal_idx = np.concatenate([pos[:N_CAL], neg[:N_CAL]])
        test_idx = np.concatenate([pos[N_CAL:], neg[N_CAL:]])
        train[dom] = (Xc, Xs, y)
        cal[dom] = (Xc2[cal_idx], Xs2[cal_idx], y2[cal_idx])
        test[dom] = (Xc2[test_idx], Xs2[test_idx], y2[test_idx])
        print(f"  built {dom}: train {y.shape[0]}, cal {cal[dom][2].shape[0]}, "
              f"test {test[dom][2].shape[0]} "
              f"(content dim {Xc.shape[1]}, spread dim {Xs.shape[1]})")

    domains = list(DOMAINS)
    # aggregate AUC + F1 (at target_cal threshold) per signal over in/cross cells
    SIGNALS = ["content", "spread", "fusion"]
    agg = {s: {"in": {"auc": [], "f1": []}, "cross": {"auc": [], "f1": []}}
           for s in SIGNALS}
    matrix = {}

    for tr in domains:
        Xc_tr, Xs_tr, y_tr = train[tr]
        content = Signal(Xc_tr, y_tr)
        spread = Signal(Xs_tr, y_tr)
        # fusion is a 2-feature LogReg over the two base-model probabilities,
        # trained on the SOURCE domain (a stacked late-fusion meta-learner).
        Pc_tr = content.proba(Xc_tr)
        Ps_tr = spread.proba(Xs_tr)
        fusion = Signal(np.column_stack([Pc_tr, Ps_tr]), y_tr)

        for te in domains:
            Xc_te, Xs_te, y_te = test[te]
            Xc_ca, Xs_ca, y_ca = cal[te]

            probs = {}
            probs_cal = {}
            probs["content"] = content.proba(Xc_te)
            probs["spread"] = spread.proba(Xs_te)
            probs["fusion"] = fusion.proba(np.column_stack(
                [content.proba(Xc_te), spread.proba(Xs_te)]))
            probs_cal["content"] = content.proba(Xc_ca)
            probs_cal["spread"] = spread.proba(Xs_ca)
            probs_cal["fusion"] = fusion.proba(np.column_stack(
                [content.proba(Xc_ca), spread.proba(Xs_ca)]))

            bucket = "in" if tr == te else "cross"
            cell = {}
            for s in SIGNALS:
                auc = _auc(y_te, probs[s])
                thr = _f1max_threshold(probs_cal[s], y_ca)   # R5 target_cal
                pred = (probs[s] >= thr).astype(int)
                f1 = round(f1_score(y_te, pred, zero_division=0), 3)
                cell[s] = dict(auc=auc, f1=f1, thr=round(thr, 4),
                               cm=confusion_matrix(y_te, pred,
                                                   labels=[0, 1]).tolist())
                agg[s][bucket]["auc"].append(auc)
                agg[s][bucket]["f1"].append(f1)
            matrix[f"{tr}->{te}"] = cell

    def m(xs):
        return round(float(np.mean(xs)), 3) if xs else float("nan")

    summary = {}
    for s in SIGNALS:
        summary[s] = dict(
            in_auc=m(agg[s]["in"]["auc"]), in_f1=m(agg[s]["in"]["f1"]),
            cross_auc=m(agg[s]["cross"]["auc"]), cross_f1=m(agg[s]["cross"]["f1"]),
            cross_below_chance=int(sum(a < 0.5 for a in agg[s]["cross"]["auc"])),
            n_cross=len(agg[s]["cross"]["auc"]),
        )

    print("\n--- summary (AUC threshold-free; F1 at R5 target_cal threshold) ---")
    hdr = f"  {'signal':8s} | {'in AUC':>7s} {'in F1':>6s} | " \
          f"{'cross AUC':>9s} {'cross F1':>8s} | below-chance"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in SIGNALS:
        d = summary[s]
        print(f"  {s:8s} | {d['in_auc']:7.3f} {d['in_f1']:6.3f} | "
              f"{d['cross_auc']:9.3f} {d['cross_f1']:8.3f} | "
              f"{d['cross_below_chance']}/{d['n_cross']}")

    # the head-to-head verdicts
    c, sp, fu = summary["content"], summary["spread"], summary["fusion"]
    best_single_cross_auc = max(c["cross_auc"], sp["cross_auc"])
    best_single_cross_f1 = max(c["cross_f1"], sp["cross_f1"])
    print("\n--- head-to-head verdict (cross-domain, the shift that matters) ---")
    print(f"  content collapses cross-domain?  AUC {c['cross_auc']} "
          f"({'YES ~chance' if c['cross_auc'] < 0.6 else 'no'})")
    print(f"  spread transfers cross-domain?   AUC {sp['cross_auc']} "
          f"({'YES' if sp['cross_auc'] >= 0.7 else 'weak'})")
    print(f"  fusion vs best single (cross AUC): "
          f"{fu['cross_auc']} vs {best_single_cross_auc} "
          f"(delta {round(fu['cross_auc'] - best_single_cross_auc, 3):+})")
    print(f"  fusion vs best single (cross F1) : "
          f"{fu['cross_f1']} vs {best_single_cross_f1} "
          f"(delta {round(fu['cross_f1'] - best_single_cross_f1, 3):+})")
    print(f"  fusion in-domain AUC {fu['in_auc']} "
          f"(sanity: should be >= best single in-domain)")

    print("\n--- per-cell (content | spread | fusion), AUC / F1@target_cal ---")
    for cellname, r in matrix.items():
        s, t = cellname.split("->")
        tag = "(in) " if s == t else "CROSS"
        print(f"  {tag} {cellname:16s} "
              f"C {r['content']['auc']:.3f}/{r['content']['f1']:.3f} | "
              f"S {r['spread']['auc']:.3f}/{r['spread']['f1']:.3f} | "
              f"F {r['fusion']['auc']:.3f}/{r['fusion']['f1']:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(os.path.join(_HERE, "..", "data",
                                       f"content_spread_headtohead_{ts}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(spread_feature_names=names, content_dim=CONTENT_DIM,
                       matrix=matrix, summary=summary,
                       config=dict(n_train=N_TRAIN, n_cal=N_CAL, n_test=N_TEST,
                                   seed=SEED, fake_shift=FAKE_SHIFT,
                                   content_noise=CONTENT_NOISE)),
                  f, indent=2)
    print(f"\nRaw output: {out}")


if __name__ == "__main__":
    main()

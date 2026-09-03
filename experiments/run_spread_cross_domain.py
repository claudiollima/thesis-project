"""
R4 — Spread-signal cross-domain transfer (the positive half of the thesis).

R1-R3 established the NEGATIVE half empirically: content-forensic detectors
collapse to ~chance across generators (mean cross-generator AUC 0.44). This
script tests the POSITIVE half: does a detector built on SPREAD-PATTERN signal
survive the shift on ITS natural axis of variation?

For content the natural threat is a new *generator* (different pixels). For
spread the natural threat is a new *domain* (different platform / topic / scale
of cascade): a meme ecosystem, a political-news ecosystem, a niche community.
If the organic-vs-coordinated STRUCTURE (Murugan et al. 2511.18733: groupthink
blending, bridge-node bottlenecks, fidelity landscape) is domain-invariant, then
a spread detector trained on one domain should transfer to another — where the
content detector, on its own axis, does not.

Honesty framing (Dr. Santos discipline):
  * Cascades are SIMULATED. This is a MECHANISM-VALIDATION experiment, not a
    real-world/in-the-wild measurement. It tests an internal-consistency claim:
    are the extracted spread features scale-invariant enough to transfer?
  * The simulator is NOT hand-tuned to the classifier. It only encodes the
    qualitative organic/coordinated contrasts from the literature at three
    genuinely different absolute SCALES. Whether transfer happens is then an
    empirical question about the feature extractor, decided by the numbers.
  * The interesting, non-tautological result is the FEATURE-GROUP contrast:
    scale-dependent features (raw follower counts, total shares, rates) should
    NOT transfer across domains, while dimensionless structural features
    (CVs, fractions, virality) should. That contrast is the actual finding.

Same protocol as R1-R3: StandardScaler + LogisticRegression, balanced sets,
fixed seeds, F1 / AUC on held-out splits, full N x N matrix.

Feature extractor: reused verbatim from research/experiments/spread_patterns.py
(the 28-feature set already documented in THESIS.md, Feb 17 + Feb 23 theory).

Author: Claudio L. Lima
Date: 2026-09-03
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# --- reuse the documented spread feature extractor -------------------------
_RESEARCH_EXP = os.path.expanduser(
    "~/.openclaw/agents/claudiollm/workspace/research/experiments"
)
sys.path.insert(0, _RESEARCH_EXP)
from spread_patterns import (  # noqa: E402
    SpreadPatternExtractor,
    ContentCascade,
    ShareEvent,
)


# ---------------------------------------------------------------------------
# Domain-parameterized cascade simulator
# ---------------------------------------------------------------------------
# Each domain fixes the ABSOLUTE scale (cascade size, follower magnitude,
# timing tempo, base verified rate). The organic-vs-coordinated STRUCTURE is
# identical across domains -- only the scale changes. That is the whole point:
# it is the mirror image of the content experiment, where the label structure
# is fixed and the *generator* changes.

DOMAINS = {
    # meme ecosystem: small, fast, low-follower
    "meme": dict(
        size=(8, 40), follower_scale=5.0, tempo_sec=90.0,
        verified_base=0.01, platform="x",
    ),
    # political-news ecosystem: large, slow, high-follower
    "news": dict(
        size=(60, 300), follower_scale=8.0, tempo_sec=3600.0,
        verified_base=0.12, platform="x",
    ),
    # niche-community ecosystem: medium, medium tempo, cross-platform prone
    "niche": dict(
        size=(20, 80), follower_scale=6.0, tempo_sec=900.0,
        verified_base=0.05, platform="forum",
    ),
}


def _simulate_cascade(rng, domain_cfg, coordinated, cid):
    """Build one ContentCascade. `coordinated=True` -> synthetic/coordinated.

    Realism: the label sets a *coordination strength* c that only SHIFTS the
    generating distributions; it does not cleanly separate them. c is drawn from
    overlapping Beta distributions (coordinated favours high c, organic low c but
    the tails cross), and every behavioural parameter interpolates on c with
    added per-account noise. This deliberately produces CLASS OVERLAP so the
    in-domain problem is honest (target AUC ~0.8, not 1.0) rather than the
    degenerate perfect-separation trap flagged for synthetic_test in R2.
    """
    lo, hi = domain_cfg["size"]
    n = int(rng.integers(lo, hi + 1))
    fscale = domain_cfg["follower_scale"]
    tempo = domain_cfg["tempo_sec"]
    plat = domain_cfg["platform"]
    base_time = datetime(2026, 1, 1, 12, 0, 0)

    # coordination strength in [0,1]; overlapping Betas -> classes bleed together
    # (moderate overlap: tuned for an honest in-domain AUC ~0.8, not 1.0 and not noise)
    if coordinated:
        c = float(rng.beta(3.0, 1.4))   # skewed high, tail reaches ~0.25
    else:
        c = float(rng.beta(1.4, 3.0))   # skewed low, tail reaches ~0.75

    # ---- inter-arrival timing: interpolate regular(coord) <-> bursty(organic) --
    # regular component (low CV) and heavy-tailed component (high CV), mixed by c
    reg = np.clip(rng.exponential(tempo * 0.25, n) + rng.normal(0, tempo * 0.05, n),
                  tempo * 0.01, None)
    short = rng.random(n) < (0.3 * (1 - c))
    burst = np.where(short, rng.exponential(tempo * 0.1, n),
                     rng.exponential(tempo * 2.0, n))
    intervals = c * reg + (1 - c) * burst
    times = np.concatenate([[0.0], np.cumsum(intervals)[: n - 1]])

    shares = []
    ids = [f"{cid}_s{i}" for i in range(n)]
    for i in range(n):
        # ---- account properties: means shift with c, moderately overlapping ----
        # age: coordinated younger (small mean); noise CV~0.55 leaves real overlap
        age_mean = c * 60 + (1 - c) * 700
        age = int(abs(rng.normal(age_mean, age_mean * 0.55)))
        # followers: coordinated smaller mu; sigma~1.0 keeps classes distinguishable
        mu = fscale - 1.6 * c
        followers = int(rng.lognormal(mu, 1.0))
        following = int(rng.lognormal(5, 1.2))
        verified = rng.random() < domain_cfg["verified_base"] * (1 - 0.85 * c)

        # ---- cascade topology (depth): direct-reshare prob rises with c --------
        if i == 0:
            depth, parent = 0, None
        elif rng.random() < (0.3 + 0.5 * c):
            # broadcast-style direct reshare from origin (shallow)
            depth, parent = 1, ids[0]
        else:
            # diffusion: parent is a recent node, depth grows
            parent = ids[int(rng.integers(max(0, i - 5), i))]
            depth = min(i, int(rng.integers(1, 6)))

        # ---- platform: organic-leaning niche cascades spread cross-platform ----
        if plat == "forum" and rng.random() < 0.25 * (1 - c):
            ev_plat = "reddit"
        else:
            ev_plat = plat

        shares.append(
            ShareEvent(
                timestamp=base_time + timedelta(seconds=float(times[i])),
                account_id=ids[i],
                account_age_days=max(1, age),
                follower_count=max(0, followers),
                following_count=max(0, following),
                is_verified=bool(verified),
                platform=ev_plat,
                parent_share_id=parent,
                depth=depth,
            )
        )

    return ContentCascade(
        content_id=cid,
        original_post_time=base_time,
        platform=plat,
        shares=shares,
        is_synthetic=coordinated,
        content_type="image",
    )


def build_domain_dataset(domain_name, n_per_class, seed):
    """Return (X, y, feature_names) for one domain, balanced."""
    rng = np.random.default_rng(seed)
    cfg = DOMAINS[domain_name]
    extractor = SpreadPatternExtractor(observation_window_hours=48)
    names = extractor.get_feature_names()

    rows, labels = [], []
    for coord, label in [(True, 1), (False, 0)]:
        for k in range(n_per_class):
            cid = f"{domain_name}_{label}_{k}"
            casc = _simulate_cascade(rng, cfg, coord, cid)
            feats = extractor.extract_all_features(casc)
            rows.append([feats[n] for n in names])
            labels.append(label)

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    # clean inf/nan (e.g. inf timing on degenerate cascades) -> large finite
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X, y, names


# ---------------------------------------------------------------------------
# Structural (scale-invariant) feature subset -- dimensionless quantities.
# Excludes raw counts / rates / absolute magnitudes that scale with domain.
# ---------------------------------------------------------------------------
STRUCTURAL = {
    "inter_share_cv", "burstiness", "peak_hour_share_fraction",
    "structural_virality", "direct_reshare_fraction", "deep_propagation_fraction",
    "account_age_cv", "new_account_fraction", "follower_cv",
    "small_account_fraction", "verified_fraction", "temporal_clustering",
    "account_age_clustering", "cross_platform_spread",
}


def fit_eval(Xtr, ytr, Xte, yte):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(Xtr), ytr)
    proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (proba >= 0.5).astype(int)
    f1 = f1_score(yte, pred, zero_division=0)
    try:
        auc = roc_auc_score(yte, proba)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(yte, pred).tolist()
    return dict(f1=round(f1, 3), auc=round(auc, 3), cm=cm)


def run_matrix(datasets, feat_idx, label):
    """datasets: {domain: (Xtr,ytr,Xte,yte)}. feat_idx: columns to use."""
    domains = list(datasets)
    matrix = {}
    in_dom, cross = [], []
    for tr in domains:
        Xtr, ytr, _, _ = datasets[tr]
        for te in domains:
            _, _, Xte, yte = datasets[te]
            res = fit_eval(Xtr[:, feat_idx], ytr, Xte[:, feat_idx], yte)
            matrix[f"{tr}->{te}"] = res
            (in_dom if tr == te else cross).append(res)
    summ = dict(
        mean_in_domain_f1=round(np.mean([r["f1"] for r in in_dom]), 3),
        mean_in_domain_auc=round(np.mean([r["auc"] for r in in_dom]), 3),
        mean_cross_f1=round(np.mean([r["f1"] for r in cross]), 3),
        mean_cross_auc=round(np.mean([r["auc"] for r in cross]), 3),
        cross_cells_below_chance=int(sum(r["auc"] < 0.5 for r in cross)),
        n_cross_cells=len(cross),
    )
    return dict(label=label, matrix=matrix, summary=summ)


def main():
    N_TRAIN = 150   # per class
    N_VAL = 60      # per class
    SEED = 20260903

    print("R4 — Spread cross-domain transfer\n" + "=" * 60)
    datasets = {}
    for i, dom in enumerate(DOMAINS):
        Xtr, ytr, names = build_domain_dataset(dom, N_TRAIN, SEED + i)
        Xva, yva, _ = build_domain_dataset(dom, N_VAL, SEED + 100 + i)
        datasets[dom] = (Xtr, ytr, Xva, yva)
        print(f"  built {dom}: train {Xtr.shape}, val {Xva.shape}")

    all_idx = list(range(len(names)))
    struct_idx = [i for i, n in enumerate(names) if n in STRUCTURAL]
    scale_idx = [i for i in all_idx if i not in struct_idx]

    runs = {
        "all_features": run_matrix(datasets, all_idx, "all 28 features"),
        "structural_only": run_matrix(datasets, struct_idx,
                                      f"structural {len(struct_idx)} (scale-invariant)"),
        "scale_only": run_matrix(datasets, scale_idx,
                                 f"scale-dependent {len(scale_idx)}"),
    }

    for key, run in runs.items():
        s = run["summary"]
        print(f"\n[{key}] {run['label']}")
        print(f"  in-domain : F1 {s['mean_in_domain_f1']} / AUC {s['mean_in_domain_auc']}")
        print(f"  cross-dom : F1 {s['mean_cross_f1']} / AUC {s['mean_cross_auc']} "
              f"({s['cross_cells_below_chance']}/{s['n_cross_cells']} below chance)")
        for cell, r in run["matrix"].items():
            tag = "  (in)" if cell.split("->")[0] == cell.split("->")[1] else "CROSS"
            print(f"    {tag} {cell:20s} F1 {r['f1']:.3f}  AUC {r['auc']:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(os.path.dirname(__file__), "..", "data",
                       f"spread_cross_domain_{ts}.json")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(feature_names=names, runs=runs,
                       config=dict(n_train=N_TRAIN, n_val=N_VAL, seed=SEED)),
                  f, indent=2)
    print(f"\nRaw output: {out}")


if __name__ == "__main__":
    main()

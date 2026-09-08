"""
R7 — Reliability-gated fusion: can a small target slice VETO a dead channel
even though it cannot RELEARN the combiner? (SAME items, SAME shift as R6)

--------------------------------------------------------------------------------
The R6 open problem, verbatim
--------------------------------------------------------------------------------
R6 closed with:

    "Naive late fusion is ACTIVELY HARMFUL under domain shift; the robust
     deployment choice is to DROP the collapsed content channel, not fuse it --
     and this is NOT the cheaply-fixable calibration artifact R5 found.
     Selective/gated fusion that can fully down-weight a dead channel (not a
     fixed source-trained combiner) is the real open design problem."

R6 also showed the obvious first fix FAILS: a target-refit 2-feature LogReg
meta-learner on 25 labels/class gives cross-domain AUC 0.670 -- still far below
spread-alone 0.805. A 2-feature combiner on 25 labels cannot reliably learn to
IGNORE a channel that is genuinely anti-correlated on the target.

--------------------------------------------------------------------------------
R7 hypothesis (the sharp, falsifiable follow-up)
--------------------------------------------------------------------------------
R6 conflated two different questions under "use the calibration set":

  (A) RELEARN the combination weights   -> needs to fit a 2-D decision surface,
                                            statistically expensive, R6 shows 25
                                            labels is not enough.
  (B) ESTIMATE per-channel RELIABILITY  -> a channel's cross-domain AUC is ONE
                                            scalar per channel; estimating whether
                                            it is > 0.5 is statistically MUCH
                                            cheaper than fitting a combiner.

R7 tests (B): use the 25-label/class target slice ONLY to estimate each channel's
reliability (its calibration-set AUC), then GATE -- veto / down-weight channels
that look dead -- WITHOUT relearning any combination. The base per-channel models
and their probabilities are frozen from the source; the gate touches only whether
(and how much) each channel is trusted.

  H1 (veto is cheap): reliability estimation from 25 labels is enough to gate a
      dead channel off, so gated fusion >= spread-alone cross-domain and >> R6
      naive/target-refit fusion. (Combiner refit fails; reliability gate does not.)
  H0 (Dr. Santos null I must be willing to accept): 25 target labels are too few
      even to estimate one AUC reliably -> the gate mis-fires (drops a good channel
      or keeps a dead one) and gated fusion is no better than just always picking
      spread. If so, the honest conclusion is "no cheap gate; you need enough
      target labels that you might as well have trained on target."

To keep myself honest, R7 also reports the ORACLE gate (channel selection using
the *test* AUC it could never see) as the ceiling, and ALWAYS-SPREAD (the trivial
"just drop content entirely, forever" policy) as the floor a real gate must beat
to justify keeping content at all.

--------------------------------------------------------------------------------
Gating strategies compared (all reuse R6's frozen per-channel Signals)
--------------------------------------------------------------------------------
  * content_only / spread_only          -- single-signal references (R6 numbers)
  * naive_fusion (source weights)       -- R6's harmful baseline, recomputed
  * refit_fusion (target 25/class)      -- R6's failed fix, recomputed
  * hard_gate    : keep channels with cal-AUC >= 0.5 + MARGIN; average their
                   probabilities (logit mean); if none pass, keep the single best
                   by cal-AUC. A dead channel is fully removed.
  * soft_gate    : weight each channel's LOGIT by w_i = max(0, cal_AUC_i - 0.5);
                   normalize weights; sum. A dead channel (cal AUC ~ 0.5) gets
                   weight ~ 0 automatically -- continuous version of the veto.
  * oracle_gate  : hard_gate but using TEST AUC (ceiling; never deployable).
  * always_spread: fixed "drop content forever" policy (floor to beat).

Same protocol as R1-R6: StandardScaler + LogisticRegression base signals, frozen
source probabilities, full 3x3 matrix, AUC primary + F1 at R5 target_cal
threshold, fixed seeds. Content and spread come from R6's build_paired... so the
items, the coupling, and the shift are IDENTICAL to R6 -- this is a clean add-on.

Author: Claudio L. Lima
Date: 2026-09-08
"""

import os
import sys

# R6's build_paired_domain_dataset seeds each item's content latent with
# abs(hash(cid)), and CPython randomizes str hashing per process. To make this
# experiment REPRODUCIBLE (same SEED -> same items -> same numbers) without
# editing the committed R6 module we import verbatim, pin PYTHONHASHSEED and
# re-exec once if it is not already fixed. This is the whole reason R6's own
# numbers drift run-to-run; R7 removes that drift.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import warnings
from datetime import datetime

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# refit_fusion fits a 2-feature LogReg on only 2*N_CAL near-collinear probability
# columns; on some cells the design is degenerate and BLAS emits overflow/invalid
# matmul warnings. The fit still converges to a usable boundary (roc_auc is
# order-invariant and handles it); silence the numerical noise, keep the result.
warnings.filterwarnings("ignore", category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reuse R6's matched-item generator and its frozen Signal (Scaler+LogReg) VERBATIM
# so R7 is a controlled add-on: identical items, coupling, and 3x3 shift.
from run_content_spread_headtohead import (  # noqa: E402
    DOMAINS,
    build_paired_domain_dataset,
    Signal,
    _f1max_threshold,
    _auc,
)

# reliability margin for the hard gate: a channel must clear 0.5 by this much on
# the calibration slice to be trusted. 0.02 is deliberately small -- we are NOT
# trying to be clever, just to veto channels that look at/below chance.
GATE_MARGIN = 0.02
EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _cal_auc(y_cal, p_cal):
    """Reliability proxy: AUC of a channel on the small target-cal slice.

    This is the ONLY thing the gate is allowed to learn from the target. One
    scalar per channel -- the whole R7 bet that this is cheaper than a combiner.
    """
    try:
        return float(roc_auc_score(y_cal, p_cal))
    except ValueError:
        return 0.5


def _combine_logits(prob_map, channels, weights):
    """Weighted mean of channel LOGITS -> back to probability.

    weights aligned with `channels`; if all weights are 0 (all channels dead by
    the gate), fall back to an equal-weight mean so we never return a constant.
    """
    w = np.asarray([weights[c] for c in channels], dtype=float)
    if not np.any(w > 0):
        w = np.ones_like(w)
    w = w / w.sum()
    Z = np.zeros_like(_logit(prob_map[channels[0]]))
    for c, wi in zip(channels, w):
        Z = Z + wi * _logit(prob_map[c])
    return _sigmoid(Z)


N_TRAIN = 150   # per class (matches R4/R5/R6)
N_CAL = 25      # per class, target calibration slice (R5 lesson, R6 value)
N_TEST = 60     # per class, held-out target test (matches R4/R5/R6)
STRATS = [
    "content_only", "spread_only",
    "naive_fusion", "refit_fusion",
    "hard_gate", "soft_gate", "oracle_gate", "always_spread",
]


def run_one(SEED):
    """Full 3x3 gated-fusion experiment for one master seed.

    Returns (summary, matrix, gate_audit). With PYTHONHASHSEED pinned this is
    deterministic in SEED, so aggregating over several SEEDs measures robustness
    of the R7 verdict rather than a single lucky/unlucky item draw.
    """
    train, cal, test = {}, {}, {}
    for i, dom in enumerate(DOMAINS):
        Xc, Xs, y, _ = build_paired_domain_dataset(dom, N_TRAIN, SEED + i)
        Xc2, Xs2, y2, _ = build_paired_domain_dataset(
            dom, N_CAL + N_TEST, SEED + 500 + i)
        pos = np.where(y2 == 1)[0]
        neg = np.where(y2 == 0)[0]
        cal_idx = np.concatenate([pos[:N_CAL], neg[:N_CAL]])
        test_idx = np.concatenate([pos[N_CAL:], neg[N_CAL:]])
        train[dom] = (Xc, Xs, y)
        cal[dom] = (Xc2[cal_idx], Xs2[cal_idx], y2[cal_idx])
        test[dom] = (Xc2[test_idx], Xs2[test_idx], y2[test_idx])

    domains = list(DOMAINS)
    agg = {s: {"in": {"auc": [], "f1": []}, "cross": {"auc": [], "f1": []}}
           for s in STRATS}
    matrix = {}
    # gate audit: did the deployable gates make the RIGHT keep/drop call per cell?
    gate_audit = {"hard_gate": {"correct": 0, "total": 0,
                                "kept_dead_content": 0, "dropped_good_spread": 0}}

    for tr in domains:
        Xc_tr, Xs_tr, y_tr = train[tr]
        content = Signal(Xc_tr, y_tr)   # frozen source content model
        spread = Signal(Xs_tr, y_tr)    # frozen source spread model

        # R6 baselines: naive (source) + refit (target 25/cls) 2-feature combiner
        Pc_tr, Ps_tr = content.proba(Xc_tr), spread.proba(Xs_tr)
        naive = Signal(np.column_stack([Pc_tr, Ps_tr]), y_tr)

        for te in domains:
            Xc_te, Xs_te, y_te = test[te]
            Xc_ca, Xs_ca, y_ca = cal[te]
            bucket = "in" if tr == te else "cross"

            # frozen per-channel probabilities on cal + test
            p_te = {"content": content.proba(Xc_te),
                    "spread": spread.proba(Xs_te)}
            p_ca = {"content": content.proba(Xc_ca),
                    "spread": spread.proba(Xs_ca)}

            # per-channel reliability estimate from the 25/cls target slice only
            rel = {c: _cal_auc(y_ca, p_ca[c]) for c in ("content", "spread")}
            rel_test = {c: _auc(y_te, p_te[c]) for c in ("content", "spread")}

            # refit fusion: relearn the 2-feature combiner on the target cal slice
            refit = Signal(np.column_stack([p_ca["content"], p_ca["spread"]]),
                           y_ca)

            # ---- assemble each strategy's TEST + CAL probabilities ----
            strat_p_te, strat_p_ca = {}, {}
            strat_p_te["content_only"] = p_te["content"]
            strat_p_ca["content_only"] = p_ca["content"]
            strat_p_te["spread_only"] = p_te["spread"]
            strat_p_ca["spread_only"] = p_ca["spread"]
            strat_p_te["always_spread"] = p_te["spread"]
            strat_p_ca["always_spread"] = p_ca["spread"]

            strat_p_te["naive_fusion"] = naive.proba(
                np.column_stack([p_te["content"], p_te["spread"]]))
            strat_p_ca["naive_fusion"] = naive.proba(
                np.column_stack([p_ca["content"], p_ca["spread"]]))

            strat_p_te["refit_fusion"] = refit.proba(
                np.column_stack([p_te["content"], p_te["spread"]]))
            strat_p_ca["refit_fusion"] = refit.proba(
                np.column_stack([p_ca["content"], p_ca["spread"]]))

            # hard gate: keep channels with cal-AUC >= 0.5 + MARGIN (logit mean);
            # if none pass, keep the single best-by-cal-AUC channel.
            kept = [c for c in ("content", "spread")
                    if rel[c] >= 0.5 + GATE_MARGIN]
            if not kept:
                kept = [max(rel, key=rel.get)]
            hw = {c: 1.0 for c in kept}
            strat_p_te["hard_gate"] = _combine_logits(p_te, kept, hw)
            strat_p_ca["hard_gate"] = _combine_logits(p_ca, kept, hw)

            # soft gate: weight each channel logit by max(0, cal_AUC - 0.5)
            sw = {c: max(0.0, rel[c] - 0.5) for c in ("content", "spread")}
            strat_p_te["soft_gate"] = _combine_logits(
                p_te, ["content", "spread"], sw)
            strat_p_ca["soft_gate"] = _combine_logits(
                p_ca, ["content", "spread"], sw)

            # oracle gate: hard gate using TEST auc (ceiling, undeployable)
            okept = [c for c in ("content", "spread")
                     if rel_test[c] >= 0.5 + GATE_MARGIN]
            if not okept:
                okept = [max(rel_test, key=rel_test.get)]
            ow = {c: 1.0 for c in okept}
            strat_p_te["oracle_gate"] = _combine_logits(p_te, okept, ow)
            strat_p_ca["oracle_gate"] = _combine_logits(p_ca, okept, ow)

            # audit the deployable hard gate against the oracle keep/drop set
            if bucket == "cross":
                ga = gate_audit["hard_gate"]
                ga["total"] += 1
                if set(kept) == set(okept):
                    ga["correct"] += 1
                if "content" in kept and "content" not in okept:
                    ga["kept_dead_content"] += 1
                if "spread" not in kept and "spread" in okept:
                    ga["dropped_good_spread"] += 1

            # ---- metrics per strategy ----
            cell = {"rel_cal": {k: round(v, 3) for k, v in rel.items()},
                    "rel_test": {k: rel_test[k] for k in rel_test},
                    "hard_kept": kept, "oracle_kept": okept}
            for s in STRATS:
                auc = _auc(y_te, strat_p_te[s])
                thr = _f1max_threshold(strat_p_ca[s], y_ca)   # R5 target_cal
                pred = (strat_p_te[s] >= thr).astype(int)
                f1 = round(f1_score(y_te, pred, zero_division=0), 3)
                cell[s] = dict(auc=auc, f1=f1)
                agg[s][bucket]["auc"].append(auc)
                agg[s][bucket]["f1"].append(f1)
            matrix[f"{tr}->{te}"] = cell

    def m(xs):
        return round(float(np.mean(xs)), 3) if xs else float("nan")

    summary = {}
    for s in STRATS:
        summary[s] = dict(
            in_auc=m(agg[s]["in"]["auc"]), in_f1=m(agg[s]["in"]["f1"]),
            cross_auc=m(agg[s]["cross"]["auc"]), cross_f1=m(agg[s]["cross"]["f1"]),
            cross_below_chance=int(sum(a < 0.5 for a in agg[s]["cross"]["auc"])),
            n_cross=len(agg[s]["cross"]["auc"]),
        )
    return summary, matrix, gate_audit


def main():
    # aggregate over several master seeds so the R7 verdict is not a single-draw
    # artifact. With PYTHONHASHSEED pinned each seed is reproducible.
    SEEDS = [20260907, 101, 202, 303, 404]

    print("R7 — reliability-gated fusion (veto vs relearn), SAME items as R6")
    print(f"    {len(SEEDS)} seeds, PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} "
          f"(deterministic) | train {N_TRAIN}/cls cal {N_CAL}/cls test {N_TEST}/cls")
    print("=" * 66)

    per_seed = []       # list of summaries
    audits = []         # list of gate_audit dicts
    first_matrix = None
    for k, sd in enumerate(SEEDS):
        summ, matr, ga = run_one(sd)
        per_seed.append(summ)
        audits.append(ga["hard_gate"])
        if k == 0:
            first_matrix = matr
        print(f"  seed {sd}: spread {summ['spread_only']['cross_auc']:.3f} | "
              f"naive {summ['naive_fusion']['cross_auc']:.3f} | "
              f"hard {summ['hard_gate']['cross_auc']:.3f} | "
              f"soft {summ['soft_gate']['cross_auc']:.3f} | "
              f"oracle {summ['oracle_gate']['cross_auc']:.3f}")

    # mean +/- std across seeds for each strategy/metric
    def stat(strat, key):
        xs = [s[strat][key] for s in per_seed]
        return float(np.mean(xs)), float(np.std(xs))

    agg_summary = {}
    for s in STRATS:
        agg_summary[s] = {k: dict(mean=round(stat(s, k)[0], 3),
                                  std=round(stat(s, k)[1], 3))
                          for k in ("in_auc", "in_f1", "cross_auc", "cross_f1")}

    print("\n--- summary across seeds: mean +/- std "
          "(AUC threshold-free; F1 at R5 target_cal thr) ---")
    hdr = (f"  {'strategy':14s} | {'in AUC':>13s} | "
           f"{'cross AUC':>13s} | {'cross F1':>13s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in STRATS:
        a = agg_summary[s]
        print(f"  {s:14s} | "
              f"{a['in_auc']['mean']:.3f}+/-{a['in_auc']['std']:.3f} | "
              f"{a['cross_auc']['mean']:.3f}+/-{a['cross_auc']['std']:.3f} | "
              f"{a['cross_f1']['mean']:.3f}+/-{a['cross_f1']['std']:.3f}")

    spm = agg_summary["spread_only"]["cross_auc"]["mean"]
    naivem = agg_summary["naive_fusion"]["cross_auc"]["mean"]
    hardm = agg_summary["hard_gate"]["cross_auc"]["mean"]
    softm = agg_summary["soft_gate"]["cross_auc"]["mean"]
    oraclem = agg_summary["oracle_gate"]["cross_auc"]["mean"]

    print("\n--- R7 verdict (cross-domain AUC, mean over seeds) ---")
    print(f"  floor  always_spread          : {spm:.3f}")
    print(f"  R6 naive_fusion               : {naivem:.3f} "
          f"(delta vs spread {naivem - spm:+.3f})")
    print(f"  hard_gate  (deployable)       : {hardm:.3f} "
          f"(delta vs spread {hardm - spm:+.3f})")
    print(f"  soft_gate  (deployable)       : {softm:.3f} "
          f"(delta vs spread {softm - spm:+.3f})")
    print(f"  oracle_gate (ceiling)         : {oraclem:.3f} "
          f"(delta vs spread {oraclem - spm:+.3f})")

    # gate audit pooled across seeds
    tot = sum(a["total"] for a in audits)
    corr = sum(a["correct"] for a in audits)
    kept_dead = sum(a["kept_dead_content"] for a in audits)
    drop_good = sum(a["dropped_good_spread"] for a in audits)
    print("\n--- hard_gate keep/drop audit vs oracle (pooled cross cells) ---")
    print(f"  correct keep/drop set : {corr}/{tot}")
    print(f"  kept a DEAD content   : {kept_dead}")
    print(f"  dropped a GOOD spread : {drop_good}")

    beat_naive = hardm > naivem + 0.02
    matches_spread = hardm >= spm - 0.02
    oracle_beats_spread = oraclem >= spm + 0.01
    print("\n--- H1 (veto is cheap) vs H0 (gating cannot beat 'drop content') ---")
    if beat_naive and matches_spread and oracle_beats_spread:
        print("  H1 SUPPORTED: reliability gate recovers spread-level cross AUC "
              "AND the oracle shows content is worth keeping when correctly gated.")
    elif beat_naive and matches_spread:
        print("  PARTIAL: the gate REPAIRS R6's harmful fusion (>> naive, ~= "
              "spread), but even the ORACLE gate cannot beat always-dropping "
              "content -> gating fixes the harm, it does not extract extra value; "
              "on this data the optimal cross-domain policy is DROP CONTENT.")
    else:
        print("  H0 SUPPORTED: the deployable gate does not even reach "
              "spread-alone; a cheap reliability veto does not exist on this data.")

    print("\n--- per-cell cross (first seed, rel_cal C/S -> hard_kept) ---")
    for cellname, r in first_matrix.items():
        s, t = cellname.split("->")
        if s == t:
            continue
        rc = r["rel_cal"]
        print(f"  CROSS {cellname:16s} relC {rc['content']:.2f} "
              f"relS {rc['spread']:.2f} -> keep {r['hard_kept']} "
              f"(oracle {r['oracle_kept']}) | "
              f"hardAUC {r['hard_gate']['auc']:.3f} "
              f"spreadAUC {r['spread_only']['auc']:.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.abspath(os.path.join(_HERE, "..", "data",
                                       f"gated_fusion_{ts}.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(seeds=SEEDS, agg_summary=agg_summary,
                       per_seed=per_seed, first_matrix=first_matrix,
                       gate_audit_pooled=dict(correct=corr, total=tot,
                                              kept_dead_content=kept_dead,
                                              dropped_good_spread=drop_good),
                       config=dict(n_train=N_TRAIN, n_cal=N_CAL, n_test=N_TEST,
                                   gate_margin=GATE_MARGIN)),
                  f, indent=2)
    print(f"\nRaw output: {out}")


if __name__ == "__main__":
    main()

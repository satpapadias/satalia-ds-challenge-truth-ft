"""Unit tests for truthclf.metrics — hand-computed reference values."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import metrics as M  # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
def test_threshold_metrics_perfect():
    yt = [1, 1, 0, 0]
    assert approx(M.accuracy(yt, yt), 1.0)
    assert approx(M.balanced_accuracy(yt, yt), 1.0)
    assert approx(M.macro_f1(yt, yt), 1.0)


def test_balanced_accuracy_handles_imbalance():
    # 9 negatives, 1 positive; predict all negative -> acc 0.9 but bal-acc 0.5
    yt = [0] * 9 + [1]
    yp = [0] * 10
    assert approx(M.accuracy(yt, yp), 0.9)
    assert approx(M.balanced_accuracy(yt, yp), 0.5)


def test_macro_f1_value():
    yt = [1, 1, 0, 0]
    yp = [1, 0, 0, 0]            # tp=1,fn=1 (pos); tn=2,fp=0 (neg)
    # F1_pos = 2*1*0.5/(1.5)=0.6667 ; F1_neg = 2*(2/2)*(2/3)/(...)=0.8
    assert approx(M.macro_f1(yt, yp), (2 / 3 + 0.8) / 2, tol=1e-4)


def test_roc_auc_reference():
    yt = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert approx(M.roc_auc(yt, s), 0.75)


def test_roc_auc_ties():
    yt = [0, 1]
    s = [0.5, 0.5]              # tie -> AUC 0.5
    assert approx(M.roc_auc(yt, s), 0.5)


def test_pr_auc_reference():
    yt = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    assert approx(M.pr_auc(yt, s), 0.8333333, tol=1e-5)


# --- pr_auc tie handling: the defect this replaced -------------------------
@pytest.mark.parametrize("yt", [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
def test_pr_auc_all_tied_is_prevalence_regardless_of_order(yt):
    """With every score tied there is exactly one operating point, so AP is the
    positive prevalence. The old per-sample implementation returned 0.833 /
    1.000 / 0.417 for these three orderings — i.e. the reported PR-AUC was
    partly a function of CSV row order."""
    assert approx(M.pr_auc(yt, [0.5] * 4), 0.5)


def test_pr_auc_is_order_invariant_under_shuffling():
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 200)
    s = np.round(rng.uniform(size=200), 1)      # coarse -> heavy ties
    base = M.pr_auc(yt, s)
    for seed in range(5):
        p = np.random.default_rng(seed).permutation(len(yt))
        assert approx(M.pr_auc(yt[p], s[p]), base, tol=1e-12)


def test_pr_auc_matches_sklearn_on_coarse_scores():
    """Pinned against the reference implementation in the tie-heavy regime this
    dataset actually occupies (score-mode emits ~17 distinct values)."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        yt = rng.integers(0, 2, 300)
        if yt.min() == yt.max():
            continue
        s = np.round(rng.uniform(size=300), 1)
        assert approx(M.pr_auc(yt, s), average_precision_score(yt, s), tol=1e-12)


def test_pr_auc_no_positives_is_nan():
    assert math.isnan(M.pr_auc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_brier():
    yt = [1, 0]
    p = [0.75, 0.25]
    assert approx(M.brier(yt, p), ((0.75 - 1) ** 2 + (0.25 - 0) ** 2) / 2)


def test_ece_perfect_is_zero():
    yt = [1, 1, 0, 0]
    p = [1.0, 1.0, 0.0, 0.0]
    assert approx(M.ece(yt, p), 0.0)


def test_reliability_curve_counts():
    yt = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.6, 0.2]
    rc = M.reliability_curve(yt, p, n_bins=5)
    assert sum(rc["count"]) == len(yt)
    assert len(rc["mean_pred"]) == 5


# --- KEEP+TEST: ECE, pinned analytically and against sklearn's binning -----
def test_ece_analytic_known_value():
    """Hand-constructed so the answer is exact.

    Ten points, all with confidence 0.9 in the predicted class (so all land in
    the single bin (0.8, 0.9]); 7 of the 10 are correct. ECE is therefore the
    one bin's gap, |0.7 - 0.9| = 0.2, at weight 10/10.
    """
    p = [0.9] * 5 + [0.1] * 5              # conf 0.9 either way; preds 1,1,1,1,1,0,0,0,0,0
    y = [1, 1, 1, 1, 0] + [0, 0, 0, 1, 1]  # 4 correct + 3 correct = 7 of 10
    assert approx(M.ece(y, p, n_bins=10), 0.2, tol=1e-12)


def test_ece_perfectly_calibrated_is_zero():
    """Confidence 0.75 on 100 points of which exactly 75 are correct."""
    p = [0.75] * 100
    y = [1] * 75 + [0] * 25
    assert approx(M.ece(y, p, n_bins=10), 0.0, tol=1e-12)


def test_ece_bins_agree_with_sklearn_calibration_curve():
    """Our reliability bins must match calibration_curve on the bins it keeps.
    calibration_curve drops empty bins and returns no counts, which is exactly
    why we keep our own — but where both report a bin, they must agree."""
    rng = np.random.default_rng(3)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(int)
    frac_sk, mean_sk = calibration_curve(y, p, n_bins=10, strategy="uniform")
    rc = M.reliability_curve(y, p, n_bins=10)
    ours = [(mp, fp) for mp, fp, c in zip(rc["mean_pred"], rc["frac_pos"], rc["count"]) if c]
    assert len(ours) == len(mean_sk)
    for (mp, fp), m_sk, f_sk in zip(ours, mean_sk, frac_sk):
        assert approx(mp, m_sk, tol=1e-12)
        assert approx(fp, f_sk, tol=1e-12)


def test_reliability_curve_reports_empty_bins_sklearn_omits():
    """The property calibration_curve cannot give us: empty bins, with counts."""
    y, p = [1, 1, 0], [0.95, 0.92, 0.91]        # everything in the top bin
    rc = M.reliability_curve(y, p, n_bins=10)
    assert len(rc["count"]) == 10 and sum(rc["count"]) == 3
    assert rc["count"][9] == 3
    assert all(c == 0 for c in rc["count"][:9])
    assert all(math.isnan(v) for v in rc["mean_pred"][:9])
    frac_sk, _ = calibration_curve(y, p, n_bins=10, strategy="uniform")
    assert len(frac_sk) == 1, "sklearn keeps only the occupied bin"


def test_ece_bin_report_exposes_effective_resolution():
    """The structural limit: confidence-of-predicted-class lives in [0.5, 1], so
    with equal-width bins over [0,1] the lower half can never be occupied."""
    rng = np.random.default_rng(5)
    p = rng.uniform(size=400)
    y = (rng.uniform(size=400) < p).astype(int)
    rep = M.ece_bin_report(y, p, n_bins=10)
    assert rep["n_bins"] == 10
    assert rep["n_bins_occupied"] <= 5
    assert all(c == 0 for c in rep["count"][:5]), "bins below conf=0.5 are unreachable"
    assert approx(rep["ece"], M.ece(y, p, n_bins=10), tol=1e-12)


def test_bootstrap_ci_brackets_point():
    yt = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    yp = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    point, lo, hi = M.bootstrap_ci(M.accuracy, yt, yp, n_boot=500, seed=0)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_symmetric_is_one():
    yt = [1, 1, 1, 1]
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]           # b discordant = c discordant -> p = 1.0
    res = M.mcnemar(yt, a, b)
    assert res["b"] == res["c"]
    assert approx(res["p_value"], 1.0)


def test_precision_recall_and_bundle():
    yt = [1, 1, 0, 0]
    yp = [1, 0, 0, 0]                       # pos: tp=1,fn=1 ; neg: tn=2,fp=1
    pr = M.precision_recall(yt, yp)
    assert approx(pr[1][0], 1.0)            # precision_True = 1/1
    assert approx(pr[1][1], 0.5)            # recall_True = 1/2
    assert approx(pr[0][0], 2 / 3)          # precision_False = 2/3
    assert approx(pr[0][1], 1.0)            # recall_False = 2/2
    b = M.metric_bundle(yt, yp, [0.9, 0.4, 0.2, 0.1])
    for k in ("accuracy", "balanced_accuracy", "macro_f1", "precision_True",
              "recall_True", "brier", "ece", "roc_auc"):
        assert k in b


def test_bootstrap_bundle_brackets_point():
    yt = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    yp = [1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
    pb = [0.8, 0.7, 0.4, 0.2, 0.3, 0.6, 0.9, 0.1, 0.7, 0.2]
    out = M.bootstrap_bundle(yt, yp, pb, n_boot=200, seed=0)
    for k, (pt, lo, hi) in out.items():
        assert lo <= pt + 1e-9 and pt <= hi + 1e-9, k


def test_bootstrap_ci_propagates_metric_errors():
    """A bug in metric_fn must surface, not be silently counted as a degenerate
    resample. This used to sit behind `except Exception: continue`."""
    def broken(*_):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        M.bootstrap_ci(broken, [1, 0, 1], [1, 0, 0], n_boot=10)


def test_bootstrap_ci_warns_when_resamples_are_dropped():
    """Single-class data makes roc_auc NaN on every draw: the interval must not
    quietly come back as a confident one."""
    yt = [1] * 8
    with pytest.warns(RuntimeWarning, match="returned NaN"):
        point, lo, hi = M.bootstrap_ci(M.roc_auc, yt, [0.1, 0.9] * 4, n_boot=20)
    assert math.isnan(lo) and math.isnan(hi)


def test_mcnemar_exact_pvalue():
    # A correct on all 10, B wrong on all 10 -> b=10, c=0
    yt = [1] * 10
    a = [1] * 10
    b = [0] * 10
    res = M.mcnemar(yt, a, b)
    assert res["b"] == 10 and res["c"] == 0
    expected = min(1.0, 2.0 * (0.5 ** 10))     # = 0.001953125
    assert approx(res["p_value"], expected, tol=1e-9)

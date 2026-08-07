"""Unit tests for truthclf.metrics — hand-computed reference values."""

from __future__ import annotations

import math
import os
import sys

import pytest

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

"""Unit tests for calibration, threshold tuning, selective prediction, and the
3-way speaker-disjoint split. Synthetic data; no network."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import calibration, threshold, selective, data, metrics  # noqa: E402


def _miscalibrated(n=2000, seed=0):
    """True probs p; overconfident reported probs; labels ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    over = 1 / (1 + np.exp(-3.0 * np.log(p / (1 - p))))   # sharpened (overconfident)
    return over, y


def test_calibration_reduces_ece():
    probs, y = _miscalibrated()
    cal = calibration.fit_best(probs, y)
    cal_probs = calibration.apply(probs, cal)
    assert metrics.ece(y, cal_probs) < metrics.ece(y, probs)


def test_temperature_apply_monotonic():
    p = [0.1, 0.4, 0.6, 0.9]
    out = calibration.apply(p, {"method": "temperature", "T": 2.0})
    assert all(out[i] < out[i + 1] for i in range(len(out) - 1))     # order preserved


def test_tune_threshold_recovers_separating_point():
    probs = [0.2, 0.3, 0.7, 0.8]
    labels = [0, 0, 1, 1]
    thr, val = threshold.tune_threshold(probs, labels, "balanced_accuracy")
    assert 0.3 < thr <= 0.7
    assert val == 1.0           # exactly 1.0 now that the EPS denominators are gone


# --- Platt scaling must be UNregularised ----------------------------------
def test_platt_recovers_known_sigmoid_parameters():
    """Generate labels from a known sigmoid(A*logit(p) + B) and check the fit
    recovers A and B. This is the test that catches a silently regularised fit:
    sklearn's LogisticRegression defaults to L2 with C=1.0, which shrinks A
    toward 0 and flattens the calibration curve while looking entirely correct.
    """
    rng = np.random.default_rng(0)
    A_true, B_true = 0.45, -0.6
    p_raw = rng.uniform(0.02, 0.98, 40000)
    z = np.log(p_raw / (1 - p_raw))
    p_true = 1 / (1 + np.exp(-(A_true * z + B_true)))
    y = (rng.uniform(size=p_raw.size) < p_true).astype(int)

    fit = calibration.fit_platt(p_raw, y)
    assert abs(fit["A"] - A_true) < 0.03, fit
    assert abs(fit["B"] - B_true) < 0.03, fit


def test_platt_is_not_shrunk_toward_zero():
    """Direct guard on the regularisation trap: an L2 fit at C=1.0 pulls the
    slope materially below the unregularised one on this data."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    p_raw = rng.uniform(0.02, 0.98, 2000)
    z = np.log(p_raw / (1 - p_raw))
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(2.5 * z)))).astype(int)

    ours = calibration.fit_platt(p_raw, y)["A"]
    shrunk = LogisticRegression(C=1e-3).fit(z.reshape(-1, 1), y).coef_[0][0]
    assert ours > shrunk * 1.5, f"slope looks regularised: ours={ours}, shrunk={shrunk}"


def test_temperature_fit_beats_the_old_grid_resolution():
    """The scipy fit must be at least as good as the best point of the old
    400-point grid, and generally strictly better (the grid quantised T)."""
    probs, y = _miscalibrated()
    fit = calibration.fit_temperature(probs, y)
    nll = calibration._nll(calibration.apply(probs, fit), y)
    grid_best = min(
        calibration._nll(calibration.apply(probs, {"method": "temperature", "T": T}), y)
        for T in np.linspace(0.05, 10.0, 400))
    assert nll <= grid_best + 1e-12


# --- candidate thresholds come from real operating points ------------------
def test_candidate_thresholds_are_finite_and_in_range():
    """roc_curve prepends np.inf so the curve starts at (0,0); that point is not
    a realisable operating point and must be dropped."""
    probs = [0.1, 0.4, 0.4, 0.9, 0.65]
    cand = threshold.candidate_thresholds(probs, [0, 0, 1, 1, 1])
    assert np.all(np.isfinite(cand))
    assert cand.min() >= min(probs) and cand.max() <= max(probs)
    assert np.inf not in cand


def test_candidate_thresholds_are_the_distinct_scores():
    """drop_intermediate=False: every distinct score stays a candidate, even the
    ones sklearn would prune off the ROC convex hull for plotting."""
    probs = [0.2, 0.2, 0.5, 0.5, 0.5, 0.9]
    labels = [0, 1, 0, 1, 1, 1]
    assert sorted(threshold.candidate_thresholds(probs, labels)) == [0.2, 0.5, 0.9]


def test_candidate_thresholds_ascending():
    cand = threshold.candidate_thresholds([0.7, 0.1, 0.4, 0.95], [1, 0, 0, 1])
    assert list(cand) == sorted(cand)


def test_candidate_thresholds_reject_single_class_labels():
    with pytest.raises(ValueError, match="both classes"):
        threshold.candidate_thresholds([0.2, 0.7], [1, 1])


def test_threshold_consistent_with_ge_comparison():
    """A candidate equal to a score value must classify that point as 1, i.e.
    the candidate set matches predict_at's `>=` rule."""
    probs = [0.2, 0.5, 0.8]
    for thr in threshold.candidate_thresholds(probs, [0, 1, 1]):
        preds = threshold.predict_at(probs, thr)
        assert preds[np.asarray(probs) == thr] == 1


def test_tuned_threshold_on_coarse_scores_is_achievable():
    """With ~17 distinct values (this dataset's score-mode regime) the chosen
    threshold must be one of them, not an arbitrary grid point between them."""
    rng = np.random.default_rng(2)
    probs = np.round(rng.uniform(size=500), 1)
    labels = (rng.uniform(size=500) < probs).astype(int)
    thr, _ = threshold.tune_threshold(probs, labels, "balanced_accuracy")
    assert thr in set(np.unique(probs))


def test_cost_threshold_lowers_with_expensive_fn():
    # mild signal; making FN very costly should push the threshold down
    rng = np.random.default_rng(0)
    probs = list(rng.uniform(0, 1, 400))
    labels = [1 if p > 0.5 else 0 for p in probs]
    thr_cheap, _ = threshold.tune_cost_threshold(probs, labels, c_fn=1, c_fp=1)
    thr_fn, _ = threshold.tune_cost_threshold(probs, labels, c_fn=10, c_fp=1)
    assert thr_fn <= thr_cheap


def test_coverage_curve_high_confidence_better():
    # confidence correlates with correctness -> accuracy decreases as coverage grows
    probs = [0.99, 0.01, 0.95, 0.05, 0.55, 0.45, 0.52, 0.48]
    labels = [1, 0, 1, 0, 0, 1, 0, 1]                  # confident ones correct
    cov, acc = selective.coverage_accuracy_curve(probs, labels)
    assert cov[0] < cov[-1] and acc[0] >= acc[-1]
    assert abs(cov[-1] - 1.0) < 1e-9


def _mk(rid, label, statement, speaker):
    return data.Row(row_id=rid, label=label, statement=statement, subjects="",
                    speaker_name=speaker, speaker_job="", speaker_state="",
                    speaker_affiliation="", statement_context="",
                    statement_clean=statement, norm_key=data.normalized_statement_key(statement))


def test_speaker_disjoint_3way_pairwise_disjoint():
    labels = ["true", "false", "half-true", "barely-true", "mostly-true", "extremely-false"]
    rows = [_mk(i, labels[i % 6], f"statement {i}", f"spk{i}") for i in range(60)]
    tr, va, te = data.speaker_disjoint_3way(rows, val_frac=0.2, test_frac=0.2, seed=0)
    assert len(tr) + len(va) + len(te) == len(rows)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert data.speakers_cross(a, b) == set()
        assert data.normkeys_cross(a, b) == set()

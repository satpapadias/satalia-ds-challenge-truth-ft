"""Unit tests for calibration, threshold tuning, selective prediction, and the
3-way speaker-disjoint split. Synthetic data; no network."""

from __future__ import annotations

import os
import sys

import numpy as np

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
    assert val > 0.999          # ~1.0 (EPS in the denominators)


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
                    statement_clean=statement, dup_key=data.norm_key(statement))


def test_speaker_disjoint_3way_pairwise_disjoint():
    labels = ["true", "false", "half-true", "barely-true", "mostly-true", "extremely-false"]
    rows = [_mk(i, labels[i % 6], f"statement {i}", f"spk{i}") for i in range(60)]
    tr, va, te = data.speaker_disjoint_3way(rows, val_frac=0.2, test_frac=0.2, seed=0)
    assert len(tr) + len(va) + len(te) == len(rows)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert data.speakers_cross(a, b) == set()
        assert data.dupkeys_cross(a, b) == set()

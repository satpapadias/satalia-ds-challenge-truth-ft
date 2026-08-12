"""Reconciliation of two predictors into one verdict.

Pure functions, so the decision rule is tested without running any agent.

The property that matters most here is that a verdict backed by a single
predictor is still a verdict, and is never presented as though both had
answered. Partial coverage is the fine-tuned predictor's normal condition, so
this is the common path rather than an edge case.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf_agents.pooling import (OK, SourceResult, logit,  # noqa: E402
                                     pool, reconcile, sigmoid)


def ok(p, pred=None):
    return SourceResult(status=OK, probability=p,
                        prediction=(1 if p >= 0.5 else 0) if pred is None else pred)


def missing(reason="FineTunedRowNotCached", status="unavailable"):
    return SourceResult(status=status, reason=reason, detail="no recorded probability")


# --- the pooling function --------------------------------------------------
def test_logit_and_sigmoid_are_inverses():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-12)


def test_saturated_probabilities_do_not_produce_infinities():
    """The recorded fine-tuned probabilities include values below 1e-7."""
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))
    assert 0.0 < pool(0.5, 0.0) < 1.0
    assert 0.0 < pool(0.5, 1.0) < 1.0


def test_w_of_one_defers_entirely_to_the_fine_tuned_model():
    assert pool(0.9, 0.2, w=1.0) == pytest.approx(0.2, abs=1e-9)


def test_w_of_zero_defers_entirely_to_zero_shot():
    assert pool(0.9, 0.2, w=0.0) == pytest.approx(0.9, abs=1e-9)


def test_equal_weight_is_the_geometric_mean_of_the_odds():
    """Log-odds pooling averages evidence, not probabilities."""
    a, b = 0.8, 0.6
    pooled = pool(a, b, w=0.5)
    odds = math.sqrt((a / (1 - a)) * (b / (1 - b)))
    assert pooled == pytest.approx(odds / (1 + odds), abs=1e-9)


def test_pooling_differs_from_averaging_the_probabilities():
    """If these agreed there would be no reason to prefer one."""
    assert pool(0.95, 0.90, w=0.5) != pytest.approx((0.95 + 0.90) / 2, abs=1e-3)


def test_pooling_interpolates_and_does_not_accumulate_evidence():
    """Two predictors that agree do not make the pool more confident.

    The weights sum to one, so this is a weighted average in log-odds space, not
    a sum. Summing would treat the predictors as independent evidence and push
    agreement towards certainty -- wrong here, since both share a base model and
    a prompt and their errors are strongly correlated.
    """
    assert pool(0.9, 0.9, w=0.5) == pytest.approx(0.9, abs=1e-9)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert 0.2 <= pool(0.2, 0.8, w=w) <= 0.8


# --- both sources ----------------------------------------------------------
def test_both_sources_pool_and_say_so():
    out = reconcile(ok(0.8), ok(0.6), w=0.5)
    assert out["status"] == "ok"
    assert out["reconciliation"]["applied"] is True
    assert out["reconciliation"]["reason"] == "both_sources"
    assert out["reconciliation"]["sources_used"] == ["zero_shot", "fine_tuned"]
    assert out["agreement"] == {"zero_shot": True, "fine_tuned": True}


def test_disagreement_is_resolved_by_the_weight_not_by_a_vote():
    out = reconcile(ok(0.95), ok(0.10), w=1.0)
    assert out["verdict"] is False
    assert out["probability"] == pytest.approx(0.10, abs=1e-6)
    assert out["agreement"] == {"zero_shot": True, "fine_tuned": False}


# --- one source: the common case ------------------------------------------
@pytest.mark.parametrize("status,reason", [
    ("unavailable", "FineTunedRowNotCached"),
    ("timeout", "PeerError"),
    ("error", "PeerError"),
    ("not_returned", "not_returned"),
])
def test_every_way_of_not_answering_takes_the_same_path(status, reason):
    """A timeout, a dead agent, a short batch and an uncovered statement all
    reach reconciliation identically, so restoring a live endpoint removes a
    condition rather than changing the logic."""
    out = reconcile(ok(0.7), missing(reason, status))
    assert out["status"] == "ok"
    assert out["verdict"] is True
    assert out["reconciliation"]["reason"] == "single_source"
    assert out["reconciliation"]["applied"] is False


def test_a_single_source_verdict_is_a_verdict_not_a_failure():
    out = reconcile(ok(0.7), missing())
    assert out["status"] == "ok"
    assert out["verdict"] is True
    assert out["probability"] == pytest.approx(0.7)


def test_the_single_source_probability_is_passed_through_unmodified():
    """Pooling did not run, so nothing may reshape the surviving probability."""
    out = reconcile(ok(0.6313), missing(), w=1.0)
    assert out["probability"] == pytest.approx(0.6313, abs=1e-9)


def test_a_missing_predictor_reads_as_null_never_as_false():
    """`false` and `did not answer` must not be the same value on the wire."""
    out = reconcile(ok(0.7), missing())
    assert out["agreement"]["fine_tuned"] is None
    assert out["sources"]["fine_tuned"]["prediction"] is None
    assert out["sources"]["fine_tuned"]["probability"] is None


def test_the_reason_names_which_predictor_was_absent_and_why():
    out = reconcile(ok(0.7), missing())
    detail = out["reconciliation"]["detail"]
    assert "fine-tuned" in detail
    assert "FineTunedRowNotCached" in detail
    assert out["sources"]["fine_tuned"]["reason"] == "FineTunedRowNotCached"


def test_applied_is_always_present_so_absence_is_never_inferred():
    """A reader who does not look for a missing key would otherwise assume
    both predictors answered."""
    for out in (reconcile(ok(0.7), ok(0.6)), reconcile(ok(0.7), missing()),
                reconcile(missing(), missing())):
        assert "applied" in out["reconciliation"]
        assert set(out["sources"]) == {"zero_shot", "fine_tuned"}
        assert "status" in out["sources"]["zero_shot"]


def test_zero_shot_alone_works_symmetrically():
    out = reconcile(missing("PeerError", "timeout"), ok(0.3))
    assert out["verdict"] is False
    assert out["reconciliation"]["sources_used"] == ["fine_tuned"]


# --- neither source --------------------------------------------------------
def test_no_source_is_the_only_case_without_a_verdict():
    out = reconcile(missing(), missing("PeerError", "error"))
    assert out["status"] == "no_verdict"
    assert out["verdict"] is None
    assert out["probability"] is None
    assert out["reconciliation"]["reason"] == "no_source"
    assert out["reconciliation"]["sources_used"] == []


# --- threshold -------------------------------------------------------------
def test_the_verdict_uses_the_threshold_it_was_given():
    assert reconcile(ok(0.55), ok(0.55), threshold=0.5)["verdict"] is True
    assert reconcile(ok(0.55), ok(0.55), threshold=0.6)["verdict"] is False


def test_verdicts_are_json_booleans_not_numpy_or_int():
    """The published response promises true/false."""
    out = reconcile(ok(0.7), ok(0.7))
    assert out["verdict"] is True
    assert isinstance(out["agreement"]["zero_shot"], bool)

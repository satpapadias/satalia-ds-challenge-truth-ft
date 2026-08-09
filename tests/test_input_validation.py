"""Predictor-boundary validation and explainer degradation detection.

Both defects these cover are harmless in a script and dangerous in a service:
a null statement renders the literal "None" into the prompt and gets scored, and
an explainer run over a degraded cache produces a complete driver distribution
built partly from p=0.5 non-measurements.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import data, explain, prompts  # noqa: E402
from truthclf.predictors.base import (InvalidPointError, LabelCountMismatch,  # noqa: E402
                                      validate_points)
from truthclf.predictors.zeroshot import ZeroShotPredictor  # noqa: E402


def mk(rid=0, statement="Taxes rose 10 percent.", speaker="jane-doe"):
    return data.Row(row_id=rid, label="false", statement=statement, subjects="",
                    speaker_name=speaker, speaker_job="", speaker_state="",
                    speaker_affiliation="", statement_context="",
                    statement_clean=statement,
                    norm_key=data.normalized_statement_key(statement or ""))


class _Client:
    def __init__(self, scores=("80",)):
        self.scores = list(scores)

    def score(self, messages_list):
        return [self.scores[i % len(self.scores)] for i in range(len(messages_list))]


def _pred(**kw):
    return ZeroShotPredictor(model="m", client=_Client(), use_logprobs=False, **kw)


# --- 3a: the null-statement path --------------------------------------------
def test_the_defect_this_prevents_is_real():
    """Without validation the prompt contains the literal string 'None'."""
    row = mk(statement=None)
    row.statement_clean = None
    assert "None" in prompts.build_user_prompt(row, "full", "score")


@pytest.mark.parametrize("statement", [None, "", "   ", "\n\t"])
def test_unusable_statement_is_rejected_by_name(statement):
    row = mk(rid=7, statement=statement)
    row.statement_clean = statement
    with pytest.raises(InvalidPointError, match="point 7 has no usable statement"):
        _pred().predict([row])


def test_a_valid_row_still_passes():
    assert _pred().predict([mk()]).n == 1


def test_validation_happens_before_any_client_call():
    class Exploding:
        def score(self, messages_list):
            raise AssertionError("client must not be called on invalid input")

    p = ZeroShotPredictor(model="m", client=Exploding(), use_logprobs=False)
    bad = mk(statement="")
    bad.statement_clean = ""
    with pytest.raises(InvalidPointError):
        p.predict([bad])


def test_label_count_mismatch_is_named():
    with pytest.raises(LabelCountMismatch, match="1 labels for 2 points"):
        _pred().predict([mk(0), mk(1)], labels=[1])


def test_validate_points_accepts_a_row_with_only_a_raw_statement():
    """statement_clean may legitimately be empty; statement is the fallback."""
    row = mk(statement="x")
    row.statement_clean = ""
    validate_points([row])


# --- the empty-batch contract -----------------------------------------------
def test_empty_batch_is_a_no_op_not_an_error():
    res = _pred().predict([])
    assert res.n == 0 and res.probs == [] and res.preds == []
    assert res.metrics is None


def test_empty_batch_with_empty_labels_no_longer_raises():
    """Previously surfaced as an opaque sklearn ValueError on empty arrays."""
    res = _pred().predict([], labels=[])
    assert res.n == 0
    assert res.metrics is None, "nothing to score, so metrics must be None"


# --- 3b: explainer degradation ----------------------------------------------
class _Degraded:
    """Predictor whose responses partly fail to parse, as on a stale cache."""
    variant = "full"

    def __init__(self, n_failures):
        self.n_failures = n_failures

    def predict(self, rows, labels=None):
        return SimpleNamespace(probs=[0.8] * len(rows),
                               parse_failures=self.n_failures, n=len(rows))


def test_any_parse_failure_fails_the_run_by_default():
    with pytest.raises(explain.DegradedPredictions, match="neutral p=0.5"):
        explain.explain(_Degraded(1), [mk(0), mk(1)], with_rationale=False)


def test_clean_run_passes_and_reports_zero():
    res = explain.explain(_Degraded(0), [mk(0), mk(1)], with_rationale=False)
    assert res["parse_failures"] == 0
    assert res["parse_failure_rate"] == 0.0
    assert res["n_predictions"] == 12          # 2 points x 6 variants


def test_tolerance_is_configurable_and_boundary_is_inclusive():
    # 1 failure out of 12 predictions = 8.33%
    explain.explain(_Degraded(1), [mk(0), mk(1)], with_rationale=False,
                    max_parse_failure_rate=1 / 12)
    with pytest.raises(explain.DegradedPredictions):
        explain.explain(_Degraded(2), [mk(0), mk(1)], with_rationale=False,
                        max_parse_failure_rate=1 / 12)


def test_rate_is_reported_in_the_aggregate():
    res = explain.explain(_Degraded(1), [mk(0), mk(1)], with_rationale=False,
                          max_parse_failure_rate=0.5)
    agg = explain.aggregate(res)
    assert agg["parse_failures"] == 1
    assert agg["parse_failure_rate"] == pytest.approx(1 / 12)


def test_predictor_without_parse_failures_is_not_blocked():
    """Test doubles that do not expose the counter must still work; the gate
    cannot fire, and the result says so with None rather than 0."""
    class Bare:
        variant = "full"

        def predict(self, rows, labels=None):
            return SimpleNamespace(probs=[0.8] * len(rows))

    res = explain.explain(Bare(), [mk()], with_rationale=False)
    assert res["parse_failures"] is None
    assert res["parse_failure_rate"] is None

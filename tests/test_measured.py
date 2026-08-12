"""Per-row reporting of whether the model actually answered.

The neutral fallback is 50/100 in score mode and 0.5 in logprob mode, and a model
that genuinely returns 50 produces exactly the same probability. So "the model
declined to judge" and "the output could not be parsed" are indistinguishable in
`probs`, and the aggregate `parse_failures` counter says how many of the latter
occurred but not which rows. `measured` closes that gap without touching `probs`.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data  # noqa: E402
from truthclf.predictors.zeroshot import ZeroShotPredictor  # noqa: E402


def rows(n=3):
    return [data.Row(row_id=i, label="true", statement=f"Claim number {i}.",
                     subjects="", speaker_name="x", speaker_job="",
                     speaker_state="", speaker_affiliation="",
                     statement_context="", statement_clean=f"Claim number {i}.",
                     norm_key=data.normalized_statement_key(f"Claim number {i}."))
            for i in range(n)]


class ScoreClient:
    def __init__(self, texts):
        self.texts = texts

    def score(self, messages_list):
        return list(self.texts)


class LogprobClient:
    def __init__(self, payloads):
        self.payloads = payloads

    def classify(self, messages_list):
        return list(self.payloads)


# --- score mode ------------------------------------------------------------
def test_a_genuine_fifty_is_measured_and_a_failure_is_not():
    """The case the field exists for: identical probabilities, different meaning."""
    pred = ZeroShotPredictor(model="m", use_logprobs=False,
                             client=ScoreClient(["50", "not a number", "50"]))
    res = pred.predict(rows(3))
    assert res.probs == [0.5, 0.5, 0.5]        # indistinguishable here ...
    assert res.measured == [True, False, True]  # ... and separable here
    assert res.parse_failures == 1


def test_measured_is_aligned_with_probs():
    pred = ZeroShotPredictor(model="m", use_logprobs=False,
                             client=ScoreClient(["90", "", "10"]))
    res = pred.predict(rows(3))
    assert len(res.measured) == len(res.probs) == res.n
    assert res.measured == [True, False, True]


def test_all_measured_when_every_row_parses():
    pred = ZeroShotPredictor(model="m", use_logprobs=False,
                             client=ScoreClient(["90", "10", "70"]))
    res = pred.predict(rows(3))
    assert res.measured == [True, True, True]
    assert res.parse_failures == 0


# --- logprob mode ----------------------------------------------------------
def test_logprob_mode_reports_measured_too():
    import math
    ok = {"top_logprobs": {"True": math.log(0.8), "False": math.log(0.2)}}
    bad = {"top_logprobs": {"Maybe": math.log(0.9)}}
    pred = ZeroShotPredictor(model="m", use_logprobs=True,
                             client=LogprobClient([ok, bad, ok]))
    res = pred.predict(rows(3))
    assert res.measured == [True, False, True]
    assert res.probs[1] == 0.5
    assert res.parse_failures == 1


# --- the guarantee that keeps the record valid -----------------------------
def test_probs_are_untouched_by_the_addition():
    """Encoding absence in `probs` would propagate through calibration,
    thresholding and every metric. It is reported alongside instead."""
    pred = ZeroShotPredictor(model="m", use_logprobs=False,
                             client=ScoreClient(["90", "oops", "10"]))
    res = pred.predict(rows(3))
    assert res.probs == [0.9, 0.5, 0.1]
    assert all(isinstance(p, float) for p in res.probs)


def test_calibration_cannot_turn_an_absent_measurement_into_a_present_one():
    from truthclf.evaluation import DecisionArtifact
    art = DecisionArtifact(
        model="m", elicitation="score",
        calibrator={"method": "platt", "A": 0.5, "B": -0.2}, threshold=0.6,
        objective="balanced_accuracy", fitted_on="unit-test split", n_val=100,
        candidate_val_nll={"temperature": 0.70, "platt": 0.66},
        nll_diff_ci=[0.04, 0.01, 0.07], selected_by="margin")
    pred = ZeroShotPredictor(model="m", use_logprobs=False, calibrator=art,
                             client=ScoreClient(["90", "oops", "10"]))
    res = pred.predict(rows(3))
    assert res.measured == [True, False, True]
    assert res.probs != [0.9, 0.5, 0.1]        # calibration did apply
    assert res.threshold == 0.6


def test_empty_batch_reports_an_empty_list_not_none():
    """None means 'not reported'; an empty batch measured nothing but did report."""
    pred = ZeroShotPredictor(model="m", use_logprobs=False, client=ScoreClient([]))
    res = pred.predict([])
    assert res.measured == []
    assert res.n == 0


def test_default_is_none_so_a_silent_predictor_is_not_read_as_all_measured():
    """A predictor that does not report must not look like one reporting success."""
    from truthclf.predictors.base import PredictionResult
    r = PredictionResult(scores=None, probs=[0.5], preds=[1], threshold=0.5,
                         parse_failures=0, n=1)
    assert r.measured is None

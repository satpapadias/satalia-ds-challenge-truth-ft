"""Tests for the shippable decision artifact (calibrator + tuned threshold).

The artifact exists so a container can reproduce the reported decision
behaviour without refitting on the validation split at start-up. These pin the
three properties that matter: it round-trips, it is opt-in, and it refuses to be
applied to a model it was not fitted for.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import calibration, evaluation as E  # noqa: E402
from truthclf.predictors.zeroshot import ZeroShotPredictor  # noqa: E402


def _artifact(model="m", method="platt", A=0.5, B=-0.2, thr=0.6,
              elicitation="score"):
    return E.DecisionArtifact(
        model=model, elicitation=elicitation,
        calibrator={"method": method, "A": A, "B": B},
        threshold=thr, objective="balanced_accuracy",
        fitted_on="unit-test split", n_val=100,
        candidate_val_nll={"temperature": 0.70, "platt": 0.66},
        nll_diff_ci=[0.04, 0.01, 0.07], selected_by="margin")


def test_roundtrip_preserves_every_field(tmp_path):
    a = _artifact()
    p = a.save(str(tmp_path / "cal.json"))
    b = E.DecisionArtifact.load(p)
    assert b.to_dict() == a.to_dict()


def test_schema_mismatch_refuses_to_load(tmp_path):
    p = tmp_path / "cal.json"
    d = _artifact().to_dict()
    d["schema"] = 999
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema"):
        E.DecisionArtifact.load(str(p))


def test_apply_matches_calibration_apply():
    a = _artifact()
    raw = [0.1, 0.4, 0.5, 0.8]
    assert a.apply(raw) == list(calibration.apply(raw, a.calibrator))


def test_decide_uses_the_tuned_threshold_not_half():
    a = _artifact(A=1.0, B=0.0, thr=0.6)
    cal, preds = a.decide([0.55, 0.65])
    assert preds == [1 if c >= 0.6 else 0 for c in cal]
    assert preds != [1 if c >= 0.5 else 0 for c in cal], "0.6 must differ from 0.5 here"


# --- the hard error -------------------------------------------------------
def test_model_mismatch_raises_rather_than_warns():
    with pytest.raises(E.CalibratorModelMismatch, match="fitted for"):
        _artifact(model="google/gemma-4-31B-it").check_model(
            "Qwen/Qwen3.5-9B", "score")


def test_predictor_rejects_a_foreign_calibrator_at_construction(tmp_path):
    p = _artifact(model="model-A").save(str(tmp_path / "cal.json"))
    with pytest.raises(E.CalibratorModelMismatch):
        ZeroShotPredictor(model="model-B", client=None, calibrator=p,
                          use_logprobs=False)


# --- elicitation is half of the identity (schema 3) -------------------------
# The bug this closes: both shipped zero-shot artifacts carry model
# "google/gemma-4-31B-it" and differ ONLY by elicitation, so a model-only check
# accepted either on either predictor. Different probability scale, different
# tuned threshold, no error.
def test_elicitation_mismatch_raises_on_a_matching_model():
    art = _artifact(model="google/gemma-4-31B-it", elicitation="score")
    with pytest.raises(E.CalibratorModelMismatch, match="elicitation|probabilities"):
        art.check_model("google/gemma-4-31B-it", "logprob")


def test_score_calibrator_is_rejected_by_a_logprob_predictor(tmp_path):
    """The exact shipped-artifact confusion, end to end at construction."""
    p = _artifact(model="google/gemma-4-31B-it",
                  elicitation="score").save(str(tmp_path / "cal.json"))
    with pytest.raises(E.CalibratorModelMismatch):
        ZeroShotPredictor(model="google/gemma-4-31B-it", client=None,
                          calibrator=p, use_logprobs=True)


def test_matching_model_and_elicitation_is_accepted():
    _artifact(model="m", elicitation="logprob").check_model("m", "logprob")


def test_check_model_requires_an_explicit_elicitation():
    """No default: the caller who forgets is the caller the check is for."""
    with pytest.raises(TypeError):
        _artifact().check_model("m")


def test_unknown_elicitation_is_rejected_at_construction():
    with pytest.raises(ValueError, match="elicitation"):
        _artifact(elicitation="logprobs")        # plural typo


def test_schema_2_artifact_will_not_load(tmp_path):
    """A schema-2 file cannot supply `elicitation`, and defaulting it would
    guess at which probability scale the file was fitted on."""
    d = _artifact().to_dict()
    d["schema"] = 2
    del d["elicitation"]
    p = tmp_path / "old.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="schema"):
        E.DecisionArtifact.load(str(p))


# --- opt-in: the raw path must be untouched --------------------------------
class _FakeClient:
    """Returns fixed scores so probabilities are deterministic."""

    def __init__(self, scores):
        self.scores = scores

    def score(self, messages_list):
        return [str(s) for s in self.scores]


def _rows():
    from truthclf import data
    return [data.Row(row_id=i, label="true", statement=f"s{i}", subjects="",
                     speaker_name="x", speaker_job="", speaker_state="",
                     speaker_affiliation="", statement_context="",
                     statement_clean=f"s{i}",
                     norm_key=data.normalized_statement_key(f"s{i}"))
            for i in range(3)]


def test_no_calibrator_returns_raw_probabilities_at_half():
    pred = ZeroShotPredictor(model="m", client=_FakeClient([90, 10, 60]),
                             use_logprobs=False)
    res = pred.predict(_rows())
    assert res.probs == [0.9, 0.1, 0.6]           # raw score/100, untouched
    assert res.preds == [1, 0, 1]
    assert res.threshold == 0.5


def test_calibrator_changes_probabilities_and_threshold():
    art = _artifact(model="m", A=0.5, B=-0.2, thr=0.6)
    pred = ZeroShotPredictor(model="m", client=_FakeClient([90, 10, 60]),
                             use_logprobs=False, calibrator=art)
    res = pred.predict(_rows())
    assert res.probs == art.apply([0.9, 0.1, 0.6])
    assert res.probs != [0.9, 0.1, 0.6]
    assert res.threshold == 0.6


def test_calibrator_is_accepted_as_a_path(tmp_path):
    p = _artifact(model="m").save(str(tmp_path / "cal.json"))
    pred = ZeroShotPredictor(model="m", client=_FakeClient([90]), calibrator=p,
                             use_logprobs=False)
    assert pred.calibrator.threshold == 0.6


def test_finetuned_predictor_passes_the_calibrator_to_its_delegate():
    from truthclf.predictors.finetuned import FinetunedPredictor
    art = _artifact(model="served-ft")
    ft = FinetunedPredictor(base_model="base", served_model="served-ft",
                            client=_FakeClient([90]), calibrator=art,
                            use_logprobs=False)
    assert ft._delegate().calibrator is art


# --- the shipped artifacts reproduce the adopted record --------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "results/curves.json")),
                    reason="record not present")
@pytest.mark.parametrize("name,elicitation", [
    ("zero_shot_decision_baseline", "logprob"),
    ("fine_tuned_decision", "logprob"),
    ("zero_shot_score_secondary", "score"),
])
def test_shipped_artifact_reproduces_the_record(name, elicitation):
    curves = json.load(open(os.path.join(ROOT, "results/curves.json")))
    art = E.DecisionArtifact.load(
        os.path.join(ROOT, f"results/calibrators/{name}.json"))
    # The two zero-shot artifacts share a model id; only this distinguishes them.
    assert art.elicitation == elicitation
    run = curves["runs"][name]
    cal, preds = art.decide(run["probs_raw"])
    assert cal == pytest.approx(run["probs_calibrated"], abs=1e-12)
    recorded = [1 if p >= run["threshold"] else 0 for p in run["probs_calibrated"]]
    assert preds == recorded

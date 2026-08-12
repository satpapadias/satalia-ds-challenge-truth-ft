"""Tests for the prediction and explanation tools.

All provider calls are mocked, so the suite stays deterministic and free.

The property under test here is that `predict` and `explain` report the same
underlying provider failure the same way. They reach the model by different
routes -- one scores a batch, the other drives occlusion through the explainer
-- so a refusal to serve the fine-tuned model is easy to translate in one and
not the other, leaving a caller to see two different errors for one cause.
"""

from __future__ import annotations

import os
import sys

import pytest
from mcp.server.mcpserver.exceptions import ToolError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf_mcp import model_tools as MT  # noqa: E402

POINTS = [{"row_id": 1, "statement": "A claim that can be checked."},
          {"row_id": 2, "statement": "Another checkable claim."}]

# The provider's response when asked to serve an adapter it cannot host. The
# text matters: it is what the translation keys on.
NOT_SERVABLE = ("Error code: 400 - {'error': {'message': 'Unable to access "
                "non-serverless model ...', 'code': 'model_not_available'}}")


class _RefusingClient:
    """A client whose every call is refused by the provider."""

    def __init__(self, message=NOT_SERVABLE):
        self.message = message

    def classify(self, messages_list):
        raise RuntimeError(self.message)

    def score(self, messages_list):
        raise RuntimeError(self.message)

    def complete(self, messages_list, max_tokens=64):
        raise RuntimeError(self.message)


@pytest.fixture
def refusing(monkeypatch):
    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: _RefusingClient())
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")


# --- the two tools must agree ----------------------------------------------
def test_predict_translates_a_refusal_to_serve_the_finetuned_model(refusing):
    with pytest.raises(ToolError) as e:
        MT.predict(model="fine_tuned", points=POINTS, fine_tuned_source="live",
                   max_live_calls=10)
    assert "FineTunedModelNotServable" in str(e.value)


def test_explain_translates_the_same_refusal_the_same_way(refusing):
    with pytest.raises(ToolError) as e:
        MT.explain(model="fine_tuned", points=POINTS, fine_tuned_source="live",
                   with_rationale=False)
    assert "FineTunedModelNotServable" in str(e.value)


def test_both_tools_name_the_same_error_class(refusing):
    def message_of(call):
        with pytest.raises(ToolError) as e:
            call()
        return str(e.value)

    p = message_of(lambda: MT.predict(model="fine_tuned", points=POINTS,
                                      fine_tuned_source="live", max_live_calls=10))
    x = message_of(lambda: MT.explain(model="fine_tuned", points=POINTS,
                                      fine_tuned_source="live", with_rationale=False))
    assert p.split(":")[0] == x.split(":")[0]


# --- an unrelated provider failure must not be relabelled ------------------
def test_an_unrelated_failure_is_not_reported_as_unservable(monkeypatch):
    monkeypatch.setattr(MT.llm, "make_client",
                        lambda *a, **k: _RefusingClient("connection reset by peer"))
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")
    with pytest.raises(Exception) as e:
        MT.predict(model="fine_tuned", points=POINTS, fine_tuned_source="live",
                   max_live_calls=10)
    assert "FineTunedModelNotServable" not in str(e.value)


def test_a_zero_shot_failure_is_not_reported_as_unservable(refusing):
    """The translation is specific to the fine-tuned model, not to the text."""
    with pytest.raises(Exception) as e:
        MT.predict(model="zero_shot", points=POINTS, max_live_calls=10)
    assert "FineTunedModelNotServable" not in str(e.value)


# --- the fine-tuned live path builds the right predictor -------------------
def test_live_finetuned_uses_the_finetuned_predictor(monkeypatch):
    """Not a zero-shot predictor aimed at the fine-tuned model id.

    The two behave identically today, but the serving configuration belongs on
    the type that owns it, so this is where a change of serving location lands.
    """
    from truthclf.predictors import FinetunedPredictor
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")
    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: _RefusingClient())
    built = MT._build_predictor("fine_tuned", "logprob")
    assert isinstance(built, FinetunedPredictor)
    assert built.served_model == MT.FINE_TUNED_MODEL


def test_zero_shot_builds_a_zero_shot_predictor(monkeypatch):
    from truthclf.predictors import ZeroShotPredictor
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")
    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: _RefusingClient())
    built = MT._build_predictor("zero_shot", "logprob")
    assert isinstance(built, ZeroShotPredictor)
    assert built.model == MT.ZERO_SHOT_MODEL


# --- explain's default -----------------------------------------------------
def test_explain_defaults_to_the_only_explainable_predictor():
    tools = {t.name: t for t in MT.server._tool_manager.list_tools()}
    assert "model" not in tools["explain"].parameters.get("required", [])
    assert tools["explain"].parameters["properties"]["model"]["default"] == "zero_shot"


def test_predict_still_requires_an_explicit_model():
    """Prediction has two working paths, so omitting the choice must not pick one."""
    tools = {t.name: t for t in MT.server._tool_manager.list_tools()}
    assert "model" in tools["predict"].parameters.get("required", [])


def test_finetuned_remains_requestable_and_fails_with_the_reason():
    """Defaulting must not hide the fine-tuned option behind a generic error."""
    with pytest.raises(ToolError) as e:
        MT.explain(model="fine_tuned", points=POINTS, fine_tuned_source="cached")
    msg = str(e.value)
    assert "CounterfactualNotAvailable" in msg
    assert "row_id" in msg


# --- the stored path is unaffected by any of this --------------------------
def test_cached_finetuned_predict_needs_no_client(monkeypatch):
    def _no_client(*a, **k):
        raise AssertionError("the stored path must not build a client")

    monkeypatch.setattr(MT.llm, "make_client", _no_client)
    out = MT.predict(model="fine_tuned",
                     points=[{"row_id": 233, "statement": "A checkable claim."}],
                     fine_tuned_source="cached")
    assert out["provenance"]["served_by"] == "cached_replay"
    assert out["provenance"]["live_calls"] == 0


# --- partial coverage of the stored fine-tuned probabilities ---------------
# Coverage is partial by nature: the recorded probabilities cover the validation
# and test splits, so a caller's own statements are usually outside them. A batch
# mixing recorded and unrecorded statements must score what it can.
def _covered_ids(n=2):
    return sorted(MT._stored_row_ids())[:n]


def test_default_refuses_a_batch_containing_an_unrecorded_statement():
    """The strict default is preserved: no silent omission."""
    points = [{"row_id": i, "statement": "x"} for i in _covered_ids()]
    points.append({"row_id": 10_000_000, "statement": "never scored"})
    with pytest.raises(ToolError) as e:
        MT.predict(model="fine_tuned", points=points, fine_tuned_source="cached")
    assert "FineTunedRowNotCached" in str(e.value)


def test_omit_scores_the_covered_rows_and_names_the_rest():
    covered = _covered_ids()
    points = [{"row_id": i, "statement": "x"} for i in covered]
    points.append({"row_id": 10_000_000, "statement": "never scored"})
    out = MT.predict(model="fine_tuned", points=points,
                     fine_tuned_source="cached", on_missing="omit")
    assert out["n"] == len(covered)
    assert out["missing_row_ids"] == [10_000_000]
    assert {p["row_id"] for p in out["predictions"]} == set(covered)
    assert out["provenance"]["served_by"] == "cached_replay"
    assert any("missing_row_ids" in w for w in out["warnings"])


def test_omit_never_substitutes_another_predictor():
    """An omitted row must be absent, not answered by something else."""
    out = MT.predict(model="fine_tuned",
                     points=[{"row_id": 10_000_000, "statement": "never scored"}],
                     fine_tuned_source="cached", on_missing="omit")
    assert out["n"] == 0
    assert out["predictions"] == []
    assert out["missing_row_ids"] == [10_000_000]
    assert out["metrics"] is None


def test_omit_filters_labels_with_the_rows_they_belong_to():
    """Metrics must never be scored against another statement's truth."""
    covered = _covered_ids()
    points = [{"row_id": i, "statement": "x"} for i in covered]
    points.append({"row_id": 10_000_000, "statement": "never scored"})
    labels = [1] * len(covered) + [0]
    out = MT.predict(model="fine_tuned", points=points, labels=labels,
                     fine_tuned_source="cached", on_missing="omit")
    # Metrics exist and were computed over the kept rows only.
    assert out["metrics"] is not None
    assert out["n"] == len(covered)


def test_default_is_unchanged_when_every_row_is_covered():
    covered = _covered_ids()
    points = [{"row_id": i, "statement": "x"} for i in covered]
    strict = MT.predict(model="fine_tuned", points=points, fine_tuned_source="cached")
    omitted = MT.predict(model="fine_tuned", points=points,
                         fine_tuned_source="cached", on_missing="omit")
    assert strict["predictions"] == omitted["predictions"]
    assert omitted["missing_row_ids"] == []

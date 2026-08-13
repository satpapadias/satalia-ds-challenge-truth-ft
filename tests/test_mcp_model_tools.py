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


# --- the stored path is bound to statement identity ------------------------
# A row_id is a position in the source dataset, not a statement. Serving a
# stored probability on a row_id match alone returns another statement's answer,
# correctly calibrated and labelled as a successful prediction. These tests use
# REAL statements for the row_ids they claim, because that is the only way the
# stored path is allowed to answer.
def _real_points(n=2):
    """(row_id, statement) pairs the stored probabilities actually belong to."""
    from truthclf import data
    clean, _ = data.clean_dataset(data.load(os.path.join(ROOT, "data.csv")), "primary")
    by_id = {r.row_id: r for r in clean}
    ids = sorted(MT._stored_identity())[:n]
    return [{"row_id": i, "statement": by_id[i].statement,
             "subjects": by_id[i].subjects,
             "speaker_name": by_id[i].speaker_name,
             "speaker_job": by_id[i].speaker_job,
             "speaker_state": by_id[i].speaker_state,
             "speaker_affiliation": by_id[i].speaker_affiliation,
             "statement_context": by_id[i].statement_context} for i in ids]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cached_finetuned_predict_needs_no_client(monkeypatch):
    def _no_client(*a, **k):
        raise AssertionError("the stored path must not build a client")

    monkeypatch.setattr(MT.llm, "make_client", _no_client)
    out = MT.predict(model="fine_tuned", points=_real_points(1),
                     fine_tuned_source="cached")
    assert out["provenance"]["served_by"] == "cached_replay"
    assert out["provenance"]["live_calls"] == 0


def test_a_novel_statement_under_an_existing_row_id_is_refused():
    """The defect this binding exists to prevent.

    row_id 0 has a stored probability. Sending a different statement under it
    previously returned row 0's probability with status ok.
    """
    with pytest.raises(ToolError) as e:
        MT.predict(model="fine_tuned",
                   points=[{"row_id": 0,
                            "statement": "Our state added 50,000 jobs last quarter."}],
                   fine_tuned_source="cached")
    msg = str(e.value)
    assert "FineTunedRowNotCached" in msg
    assert "DIFFERENT statement" in msg


def test_a_novel_statement_under_an_existing_row_id_is_omitted_not_answered():
    out = MT.predict(model="fine_tuned",
                     points=[{"row_id": 0,
                              "statement": "Our state added 50,000 jobs last quarter."}],
                     fine_tuned_source="cached", on_missing="omit")
    assert out["n"] == 0
    assert out["predictions"] == []
    assert out["missing_row_ids"] == [0]
    assert any("different statement" in w for w in out["warnings"])


def test_the_genuine_statement_for_that_row_id_still_answers():
    """The binding must not break the path it protects."""
    out = MT.predict(model="fine_tuned", points=_real_points(1),
                     fine_tuned_source="cached")
    assert out["n"] == 1
    assert out["missing_row_ids"] == []
    assert 0.0 <= out["predictions"][0]["prob"] <= 1.0


def test_default_refuses_a_batch_containing_an_unrecorded_statement():
    """The strict default is preserved: no silent omission."""
    points = _real_points(2) + [{"row_id": 10_000_000, "statement": "never scored"}]
    with pytest.raises(ToolError) as e:
        MT.predict(model="fine_tuned", points=points, fine_tuned_source="cached")
    assert "FineTunedRowNotCached" in str(e.value)


def test_omit_scores_the_covered_rows_and_names_the_rest():
    real = _real_points(2)
    points = real + [{"row_id": 10_000_000, "statement": "never scored"}]
    out = MT.predict(model="fine_tuned", points=points,
                     fine_tuned_source="cached", on_missing="omit")
    assert out["n"] == len(real)
    assert out["missing_row_ids"] == [10_000_000]
    assert {p["row_id"] for p in out["predictions"]} == {p["row_id"] for p in real}
    assert out["provenance"]["served_by"] == "cached_replay"


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
    real = _real_points(2)
    points = real + [{"row_id": 10_000_000, "statement": "never scored"}]
    out = MT.predict(model="fine_tuned", points=points, labels=[1] * len(real) + [0],
                     fine_tuned_source="cached", on_missing="omit")
    assert out["metrics"] is not None
    assert out["n"] == len(real)


def test_default_is_unchanged_when_every_row_is_covered():
    real = _real_points(2)
    strict = MT.predict(model="fine_tuned", points=real, fine_tuned_source="cached")
    omitted = MT.predict(model="fine_tuned", points=real,
                         fine_tuned_source="cached", on_missing="omit")
    assert strict["predictions"] == omitted["predictions"]
    assert omitted["missing_row_ids"] == []


# --- provenance must report measurements, not assumptions -------------------
# cache_hits / live_calls were previously hardcoded to (0, len(rows)) on the live
# path, so a request served entirely from the response cache still reported a
# live call per row. The field claimed a spend that had not happened.
class _CountingClient(_RefusingClient):
    """A client that answers and counts only the calls that reached a provider."""

    def __init__(self, n_api_calls=0):
        super().__init__()
        self.n_api_calls = n_api_calls

    def classify(self, messages_list):
        # Simulate every response coming from cache: no counter movement.
        return [{"top_logprobs": {"True": -0.1, "False": -2.0}} for _ in messages_list]


def test_a_fully_cached_request_reports_no_live_calls(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")
    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: _CountingClient())
    out = MT.predict(model="zero_shot", points=POINTS, max_live_calls=10)
    assert out["provenance"]["live_calls"] == 0
    assert out["provenance"]["cache_hits"] == len(POINTS)


def test_live_calls_reflect_the_client_counter(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")

    class OneMiss(_CountingClient):
        def classify(self, messages_list):
            self.n_api_calls += 1        # one row missed, the rest were cached
            return [{"top_logprobs": {"True": -0.1, "False": -2.0}}
                    for _ in messages_list]

    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: OneMiss())
    out = MT.predict(model="zero_shot", points=POINTS, max_live_calls=10)
    assert out["provenance"]["live_calls"] == 1
    assert out["provenance"]["cache_hits"] == len(POINTS) - 1


def test_a_client_that_does_not_count_reports_null_not_zero(monkeypatch):
    """None means 'not reported'; 0 would mean 'measured none'."""
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-used")

    class Uncounted:
        def classify(self, messages_list):
            return [{"top_logprobs": {"True": -0.1, "False": -2.0}}
                    for _ in messages_list]

    monkeypatch.setattr(MT.llm, "make_client", lambda *a, **k: Uncounted())
    out = MT.predict(model="zero_shot", points=POINTS, max_live_calls=10)
    assert out["provenance"]["live_calls"] is None
    assert out["provenance"]["cache_hits"] is None

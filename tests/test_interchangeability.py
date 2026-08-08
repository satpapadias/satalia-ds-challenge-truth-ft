"""The two predictors must be swappable, and explain() must accept either.

The spec requires Component 2's predict to have "the same signature and
behavior as Component 1's predict, so the two predictors are interchangeable",
and explain(model, ...) to work with either. Until now the suite only exercised
explain() with hand-rolled stubs, and the two predict() signatures differed in
their first parameter name — positionally compatible, but `predict(points=...)`
raised TypeError on the zero-shot predictor.
"""

from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import data, explain  # noqa: E402
from truthclf.predictors.base import Predictor  # noqa: E402
from truthclf.predictors.finetuned import FinetunedPredictor  # noqa: E402
from truthclf.predictors.zeroshot import ZeroShotPredictor  # noqa: E402


def mk(rid=0, statement="Taxes rose 10 percent.", speaker="jane-doe"):
    return data.Row(row_id=rid, label="false", statement=statement,
                    subjects="taxes$economy", speaker_name=speaker, speaker_job="",
                    speaker_state="", speaker_affiliation="republican",
                    statement_context="a TV interview", statement_clean=statement,
                    norm_key=data.normalized_statement_key(statement))


class FakeClient:
    """Deterministic stand-in for both elicitation modes."""

    def __init__(self, score="80"):
        self.score = score

    def score(self, messages_list):            # noqa: A003 - mirrors the real API
        return [self._score] * len(messages_list)

    def classify(self, messages_list):
        return [{"text": "True",
                 "top_logprobs": {"True": -0.2, "False": -1.7}}] * len(messages_list)

    def complete(self, messages_list, max_tokens=64):
        return ["The specific figure in the claim is exaggerated."] * len(messages_list)


class _ScoreClient(FakeClient):
    def __init__(self, value="80"):
        self._score = value

    def score(self, messages_list):
        return [self._score] * len(messages_list)


# --- signature parity ------------------------------------------------------
def test_predict_signatures_are_identical():
    z = inspect.signature(ZeroShotPredictor.predict)
    f = inspect.signature(FinetunedPredictor.predict)
    zp = [p for p in z.parameters][1:]
    fp = [p for p in f.parameters][1:]
    assert zp == fp == ["points", "labels"], (zp, fp)


def test_predict_is_callable_by_keyword_on_both():
    """A caller swapping one predictor for the other must need no code change."""
    rows = [mk()]
    zs = ZeroShotPredictor(model="m", client=_ScoreClient(), use_logprobs=False)
    ft = FinetunedPredictor(base_model="b", served_model="m",
                            client=_ScoreClient(), use_logprobs=False)
    for pred in (zs, ft):
        res = pred.predict(points=rows)          # keyword, not positional
        assert res.n == 1


def test_fine_tune_returns_a_handle_attribute():
    ft = FinetunedPredictor(base_model="b", served_model="m")
    assert hasattr(ft, "output_name")
    assert "training_dataset" in inspect.signature(FinetunedPredictor.fine_tune).parameters


def test_both_predictors_satisfy_the_predictor_interface():
    assert issubclass(ZeroShotPredictor, Predictor)
    assert issubclass(FinetunedPredictor, Predictor)


# --- explain() accepts either REAL predictor -------------------------------
@pytest.fixture
def points():
    return [mk(0, speaker="alice"), mk(1, "Jobs grew fast.", speaker="bob")]


def _real_predictors():
    return {
        "zero-shot": ZeroShotPredictor(model="m", client=_ScoreClient("80"),
                                       use_logprobs=False),
        "fine-tuned": FinetunedPredictor(base_model="b", served_model="m",
                                         client=_ScoreClient("80"),
                                         use_logprobs=False),
    }


@pytest.mark.parametrize("name", ["zero-shot", "fine-tuned"])
def test_explain_accepts_each_real_predictor(name, points):
    model = _real_predictors()[name]
    res = explain.explain(model, points, with_rationale=False)
    assert len(res["per_point"]) == len(points)
    for pp in res["per_point"]:
        assert set(explain.OCCLUSION_FIELDS) <= set(pp["occlusion"])
        assert pp["pred"] in (0, 1)


@pytest.mark.parametrize("name", ["zero-shot", "fine-tuned"])
def test_explain_returns_metrics_with_labels_for_each_real_predictor(name, points):
    model = _real_predictors()[name]
    res = explain.explain(model, points, labels=[1, 1], with_rationale=False)
    assert "metrics" in res and "accuracy" in res["metrics"]


@pytest.mark.parametrize("name", ["zero-shot", "fine-tuned"])
def test_explain_omits_metrics_without_labels_for_each_real_predictor(name, points):
    res = explain.explain(_real_predictors()[name], points, with_rationale=False)
    assert "metrics" not in res


def test_explain_batches_all_occlusion_variants_into_one_predict_call(points):
    """Efficiency requirement: a set of N points must not become N*6 calls."""
    calls = []

    class Counting(ZeroShotPredictor):
        def predict(self, points, labels=None):
            calls.append(len(points))
            return SimpleNamespace(probs=[0.8] * len(points))

    explain.explain(Counting(model="m", client=_ScoreClient(), use_logprobs=False),
                    points, with_rationale=False)
    assert len(calls) == 1, f"expected one batched predict(), got {len(calls)}"
    assert calls[0] == len(points) * 6, "base + 4 single-field ablations + all-metadata"

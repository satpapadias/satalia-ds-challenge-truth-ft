"""Unit tests for prompt assembly and the zero-shot predictor (no network)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import data, prompts                                      # noqa: E402
from truthclf.predictors import ZeroShotPredictor, parse_score         # noqa: E402
from truthclf.predictors.zeroshot import prob_from_logprobs            # noqa: E402


def mk(statement="Taxes went up 10 percent.", **kw):
    base = dict(
        row_id=0, label="false", statement=statement, subjects="taxes$economy",
        speaker_name="jane-doe", speaker_job="", speaker_state="",
        speaker_affiliation="republican", statement_context="a TV interview",
    )
    base.update(kw)
    base["statement_clean"] = data.clean_text(base["statement"])
    base["norm_key"] = data.normalized_statement_key(base["statement"])
    return data.Row(**base)


def tl(p_true, p_false):
    """Build a first-token logprob dict for given True/False probabilities."""
    return {"True": math.log(p_true), "False": math.log(p_false)}


# --- parsing & probability mapping -----------------------------------------
def test_parse_score_variants():
    assert parse_score("75") == 75
    assert parse_score("Score: 80") == 80
    assert parse_score("The truthfulness score is 90.") == 90
    assert parse_score("85/100") == 85
    assert parse_score("120") == 100
    assert parse_score("no number here") is None
    assert parse_score("") is None


def test_prob_from_logprobs():
    assert abs(prob_from_logprobs(tl(0.9, 0.1)) - 0.9) < 1e-9
    assert abs(prob_from_logprobs({"True": math.log(0.4), "False": math.log(0.4)}) - 0.5) < 1e-9
    # leading-space / casing variants are recognised
    assert prob_from_logprobs({" true": math.log(0.8)}) > 0.5
    assert prob_from_logprobs({" False": math.log(0.8)}) < 0.5
    # neither present -> None (caller applies fallback)
    assert prob_from_logprobs({"maybe": math.log(0.5)}) is None


# --- prompt assembly --------------------------------------------------------
def test_unknown_substitution_for_missing_fields():
    block = prompts.assemble_metadata(mk(speaker_job="", speaker_state=""))
    assert "Speaker's job: [unknown]" in block
    assert "Speaker's home state: [unknown]" in block
    assert "Speaker's affiliation: republican" in block
    assert "Subjects: taxes, economy" in block


def test_statement_only_excludes_metadata():
    p = prompts.build_user_prompt(mk(), "statement_only", "decision")
    assert "Speaker" not in p and "Subjects" not in p
    assert mk().statement_clean in p


def test_full_includes_metadata():
    p = prompts.build_user_prompt(mk(), "full", "decision")
    assert "Speaker: jane-doe" in p
    assert "Context: a TV interview" in p


def test_modes_select_system_and_cue():
    dec = prompts.build_messages(mk(), "full", "decision")
    sco = prompts.build_messages(mk(), "full", "score")
    assert "True or False" in dec[0]["content"]
    assert dec[1]["content"].rstrip().endswith("Answer (True or False):")
    assert "0 to 100" in sco[0]["content"]
    assert sco[1]["content"].rstrip().endswith("Truthfulness score (0-100):")


def test_unknown_variant_and_mode_raise():
    for bad in [dict(variant="bogus", mode="decision"), dict(variant="full", mode="bogus")]:
        try:
            prompts.build_user_prompt(mk(), **bad)
        except ValueError:
            continue
        assert False, f"expected ValueError for {bad}"


# --- predictor (mock client, no network) ------------------------------------
class FakeClient:
    """Deterministic stand-in; no network. Returns canned classify/score data."""

    def __init__(self, classify_replies, score_replies=None):
        self.classify_replies = classify_replies
        self.score_replies = score_replies or []

    def classify(self, messages_list):
        return [{"text": "", "top_logprobs": r} for r in self.classify_replies]

    def score(self, messages_list):
        return list(self.score_replies)


def test_predictor_requires_client():
    try:
        ZeroShotPredictor(model="m", client=None).predict([mk()])
    except RuntimeError:
        return
    assert False, "expected RuntimeError when no client configured"


def test_predictor_default_is_logprobs():
    # the default elicitation is logprob/decision mode (the reported baseline)
    rows = [mk(statement="A"), mk(statement="B"), mk(statement="C")]
    client = FakeClient([tl(0.9, 0.1), tl(0.2, 0.8), {}])           # third -> 0.5 fallback
    res = ZeroShotPredictor(model="m", client=client, seed=0).predict(rows)
    assert res.parse_failures == 1
    assert abs(res.probs[0] - 0.9) < 1e-9
    assert res.preds[0] == 1 and res.preds[1] == 0
    assert res.scores is None                        # no 0-100 score in logprob path


def test_predictor_score_mode_secondary():
    # score-mode is the documented secondary (continuous probabilities)
    rows = [mk(statement="A"), mk(statement="B"), mk(statement="C")]
    labels = [1, 0, 1]
    client = FakeClient([], score_replies=["90", "10", "garbage"])   # third -> neutral 50
    res = ZeroShotPredictor(model="m", client=client, seed=0,
                            use_logprobs=False).predict(rows, labels=labels)
    assert res.n == 3
    assert res.parse_failures == 1
    assert res.scores == [90, 10, 50]               # neutral fallback applied
    assert res.probs[0] == 0.9
    assert res.preds == [1, 0, 1]                    # score>=50 -> True (50 -> True)
    assert res.metrics is not None and "brier" in res.metrics

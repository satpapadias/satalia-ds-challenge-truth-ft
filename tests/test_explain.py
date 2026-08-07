"""Unit tests for the explainer (occlusion + rationale + cross-check), using a
deterministic mocked predictor. No network."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import data, explain  # noqa: E402


def mk(rid=0, statement="Taxes rose 10 percent.", speaker="jane-doe"):
    return data.Row(row_id=rid, label="false", statement=statement, subjects="taxes$economy",
                    speaker_name=speaker, speaker_job="", speaker_state="",
                    speaker_affiliation="republican", statement_context="a TV interview",
                    statement_clean=statement, norm_key=data.normalized_statement_key(statement))


class SpeakerDrivenPredictor:
    """Prob depends only on speaker_name presence -> removing speaker flips it."""
    variant = "full"

    def predict(self, rows):
        return SimpleNamespace(probs=[0.85 if r.speaker_name else 0.20 for r in rows])

    def rationale(self, rows):
        return ["The speaker is a well-known politician." for _ in rows]


class StatementClaimPredictor:
    """Same speaker-driven behaviour, but the rationale talks about the claim ->
    rationalization: occlusion says 'speaker', rationale says 'statement'."""
    variant = "full"

    def predict(self, rows):
        return SimpleNamespace(probs=[0.85 if r.speaker_name else 0.20 for r in rows])

    def rationale(self, rows):
        return ["The specific percentage figure in the claim is exaggerated." for _ in rows]


def test_occlusion_detects_driver_and_flip():
    res = explain.explain(SpeakerDrivenPredictor(), [mk()], with_rationale=True)
    pp = res["per_point"][0]
    assert pp["pred"] == 1                                  # base 0.85 -> True
    occ = pp["occlusion"]
    assert occ["speaker_name"]["flip"] == 1                 # removing speaker flips
    assert abs(occ["speaker_name"]["delta"] - 0.65) < 1e-6
    assert occ["speaker_affiliation"]["flip"] == 0          # other fields don't move it
    assert pp["driver"] == "speaker_name"


def test_cross_check_agreement_and_disagreement():
    agree = explain.explain(SpeakerDrivenPredictor(), [mk()])["per_point"][0]
    assert "speaker" in agree["rationale_refs"]
    assert agree["agree"] is True                           # occlusion+rationale both speaker

    diss = explain.explain(StatementClaimPredictor(), [mk()])["per_point"][0]
    assert diss["driver"] == "speaker_name"                 # occlusion: speaker drove it
    assert diss["agree"] is False                           # rationale cited the claim


def test_metrics_returned_with_labels():
    pts = [mk(0, speaker="a"), mk(1, speaker="")]           # preds: True, False
    res = explain.explain(SpeakerDrivenPredictor(), pts, labels=[1, 0])
    assert "metrics" in res and "accuracy" in res["metrics"]
    assert res["metrics"]["accuracy"] == 1.0


def test_aggregate_field_table_and_agreement():
    pts = [mk(i, speaker=f"spk{i}") for i in range(5)]
    res = explain.explain(SpeakerDrivenPredictor(), pts)
    agg = explain.aggregate(res)
    tbl = agg["field_table"].set_index("removed_field")
    assert tbl.loc["speaker_name", "flip_rate"] == 1.0      # speaker removal always flips
    assert tbl.loc["speaker_affiliation", "flip_rate"] == 0.0
    assert agg["rationale_occlusion_agreement_rate"] == 1.0
    assert agg["n"] == 5


def test_works_without_rationale():
    res = explain.explain(SpeakerDrivenPredictor(), [mk()], with_rationale=False)
    pp = res["per_point"][0]
    assert pp["rationale"] is None and pp["agree"] is None
    assert pp["driver"] == "speaker_name"

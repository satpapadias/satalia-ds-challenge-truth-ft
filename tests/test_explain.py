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


# ---------------------------------------------------------------------------
# Degenerate occlusion
# ---------------------------------------------------------------------------
# When no occluded field moves the probability, occlusion has measured nothing
# about that point. Attributing it to `statement` would assert a finding the
# measurement does not contain, and because `statement` is simultaneously the
# fallback rationale reference, such points would also score as agreement --
# absence on both sides counted as concordance.
class FlatPredictor:
    """Returns the same probability whatever is removed."""
    variant = "full"

    def __init__(self, p=0.5, rationale="No basis to judge this claim."):
        self.p = p
        self._rationale = rationale

    def predict(self, rows):
        return SimpleNamespace(probs=[self.p] * len(rows), parse_failures=0,
                               n=len(rows))

    def rationale(self, rows):
        return [self._rationale for _ in rows]


class SmallMovePredictor:
    """Moves the probability, but by less than driver_eps."""
    variant = "full"

    def predict(self, rows):
        return SimpleNamespace(probs=[0.62 if r.speaker_name else 0.60 for r in rows],
                               parse_failures=0, n=len(rows))

    def rationale(self, rows):
        return ["The claim cites a specific figure." for _ in rows]


def test_flat_occlusion_is_undetermined_not_statement():
    pp = explain.explain(FlatPredictor(), [mk()])["per_point"][0]
    assert all(pp["occlusion"][f]["delta"] == 0.0 for f in explain.OCCLUSION_FIELDS)
    assert pp["driver"] == "undetermined"


def test_agreement_is_not_scored_when_the_driver_is_undetermined():
    """The rationale still exists; there is simply no measured driver to match."""
    pp = explain.explain(FlatPredictor(), [mk()])["per_point"][0]
    assert pp["rationale"] is not None
    assert pp["rationale_refs"] is not None
    assert pp["agree"] is None


def test_a_flat_point_at_any_probability_is_undetermined():
    """0.5 is the neutral fallback value, so a flat run at 0.5 is the case most
    likely to be mistaken for a parse failure. Flatness is what matters, not the
    level."""
    for p in (0.0, 0.2, 0.5, 0.8, 1.0):
        pp = explain.explain(FlatPredictor(p=p), [mk()])["per_point"][0]
        assert pp["driver"] == "undetermined", p


def test_movement_below_driver_eps_is_still_statement_not_undetermined():
    """`statement` keeps its meaning: some field moved it, none by enough.
    Undetermined is a different claim -- nothing moved it at all."""
    pp = explain.explain(SmallMovePredictor(), [mk()], driver_eps=0.05)["per_point"][0]
    assert abs(pp["occlusion"]["speaker_name"]["delta"]) == 0.02
    assert pp["driver"] == "statement"
    assert pp["agree"] is not None


def test_a_measurable_driver_is_unaffected():
    pp = explain.explain(SpeakerDrivenPredictor(), [mk()])["per_point"][0]
    assert pp["driver"] == "speaker_name"
    assert pp["agree"] is True


def test_aggregate_reports_the_undetermined_share_and_agreement_denominator():
    points = [mk(rid=i) for i in range(4)]
    flat = explain.aggregate(explain.explain(FlatPredictor(), points))
    assert flat["driver_distribution"] == {"undetermined": 4}
    assert flat["n_undetermined"] == 4
    assert flat["undetermined_rate"] == 1.0
    # No point had a measurable driver, so the rate has no denominator.
    assert flat["n_agreement_scored"] == 0
    assert flat["rationale_occlusion_agreement_rate"] is None


def test_undetermined_points_do_not_dilute_the_agreement_rate():
    """Two points that agree, two with nothing to agree about, must report 1.0
    over two -- not 0.5 over four, and not 1.0 over four."""
    class Mixed:
        variant = "full"

        def predict(self, rows):
            # Rows arrive as 6 variants per point, in point order.
            probs = []
            for i, r in enumerate(rows):
                if i < 6:                      # point 0: speaker-driven
                    probs.append(0.85 if r.speaker_name else 0.20)
                else:                          # point 1: flat
                    probs.append(0.5)
            return SimpleNamespace(probs=probs, parse_failures=0, n=len(rows))

        def rationale(self, rows):
            return ["The speaker is a well-known politician." for _ in rows]

    agg = explain.aggregate(explain.explain(Mixed(), [mk(rid=0), mk(rid=1)]))
    assert agg["driver_distribution"] == {"speaker_name": 1, "undetermined": 1}
    assert agg["n_agreement_scored"] == 1
    assert agg["rationale_occlusion_agreement_rate"] == 1.0

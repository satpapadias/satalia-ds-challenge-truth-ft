"""Tests for the JSON to Row conversion used by the MCP servers.

The central property is test_built_row_matches_data_load_exactly: a Row built
from tool arguments must be identical to the Row the CSV loader produces from
the same input. Every other behaviour in the MCP layer depends on that holding.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data  # noqa: E402
from truthclf.predictors.base import (InvalidPointError,  # noqa: E402
                                      LabelCountMismatch)
from truthclf_mcp import adapter  # noqa: E402
from truthclf_mcp.errors import BatchTooLarge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data.csv")

# The metadata columns a Point carries, plus the two derived fields the adapter
# must reproduce. Named here so a new Row field makes this test fail loudly
# rather than silently going unchecked.
_POINT_KEYS = ("statement",) + adapter.METADATA_FIELDS


def _point_from(row: data.Row) -> dict:
    """Serialise a Row to the wire shape a client would send."""
    return {"row_id": row.row_id, **{k: getattr(row, k) for k in _POINT_KEYS}}


# --- the headline equality -------------------------------------------------
@pytest.mark.skipif(not os.path.exists(DATA), reason="data.csv not present")
def test_built_row_matches_data_load_exactly():
    """A Row built from tool arguments must equal the one data.load builds.

    Row defaults statement_clean and norm_key to empty strings, and prompt
    assembly uses `statement_clean or statement`. A conversion that omits
    statement_clean therefore sends the raw statement to the model: every
    probability shifts, nothing raises, and results stop matching the recorded
    evaluation. This test is what makes that regression visible.
    """
    loaded = data.load(DATA)
    # A spread across the file rather than the first few: cleaning only fires on
    # rows with curly quotes, dashes or stripped contractions (~5% of rows).
    sample = loaded[::311]
    assert len(sample) > 20, "sample too small to be meaningful"

    for original in sample:
        built = adapter.to_row(_point_from(original), index=original.row_id,
                               label=original.label)
        assert built == original, (
            f"row {original.row_id} differs:\n"
            f"  built    = {built}\n"
            f"  data.load= {original}")


@pytest.mark.skipif(not os.path.exists(DATA), reason="data.csv not present")
def test_the_derived_fields_are_the_ones_that_would_break():
    """Confirm the equality above can actually fail.

    Text cleaning only changes rows containing curly quotes, dashes or stripped
    contractions. Restricted to rows where cleaning is a no-op, the test above
    would pass even for a conversion that skipped it entirely, so it is checked
    here against rows that cleaning demonstrably changes.
    """
    loaded = data.load(DATA)
    differing = [r for r in loaded if r.statement_clean != r.statement]
    assert differing, "no row in data.csv is changed by clean_text -- the test above proves nothing"

    for original in differing[:25]:
        naive = dataclasses.replace(original, statement_clean="", norm_key="")
        built = adapter.to_row(_point_from(original), index=original.row_id,
                               label=original.label)
        assert built == original
        assert built != naive


def test_statement_clean_is_populated_for_a_synthetic_point():
    p = {"statement": "They dont know what theyre talking about — really."}
    row = adapter.to_row(p, index=0)
    assert row.statement_clean == data.clean_text(p["statement"])
    assert row.statement_clean != p["statement"]
    assert row.norm_key == data.normalized_statement_key(p["statement"])


# --- defaults and identity -------------------------------------------------
def test_missing_metadata_defaults_to_empty_not_an_error():
    """The CSV loader yields an empty string for an absent column and prompt
    assembly renders that as [unknown], so absent metadata is a defined input."""
    row = adapter.to_row({"statement": "x"}, index=7)
    for f in adapter.METADATA_FIELDS:
        assert getattr(row, f) == ""
    assert row.row_id == 7


def test_explicit_row_id_wins_over_the_index():
    assert adapter.to_row({"statement": "x", "row_id": 4242}, index=0).row_id == 4242


def test_label_placeholder_is_empty_and_y_would_raise():
    """Records why the empty label placeholder is safe: prediction and
    explanation never call y(), which would raise on an empty label."""
    row = adapter.to_row({"statement": "x"}, index=0)
    assert row.label == ""
    with pytest.raises(KeyError):
        row.y("primary")


def test_to_rows_assigns_sequential_ids():
    rows = adapter.to_rows([{"statement": "a"}, {"statement": "b"}])
    assert [r.row_id for r in rows] == [0, 1]


# --- labels ----------------------------------------------------------------
def test_binary_labels_pass_through():
    assert adapter.coerce_labels([0, 1, 1]) == [0, 1, 1]


def test_six_way_labels_go_through_binarize_not_a_local_table():
    labels = list(data.LABELS_6)
    assert adapter.coerce_labels(labels) == [data.binarize(x, "primary") for x in labels]
    assert adapter.coerce_labels(labels) == [1, 1, 1, 0, 0, 0]


def test_sensitivity_scheme_moves_half_true():
    assert adapter.coerce_labels(["half-true"], "primary") == [1]
    assert adapter.coerce_labels(["half-true"], "sensitivity") == [0]


def test_labels_none_stays_none():
    assert adapter.coerce_labels(None) is None


@pytest.mark.parametrize("bad", [[2], [-1], ["mostly true"], [None], [1.0], [True]])
def test_unusable_labels_raise(bad):
    with pytest.raises(ValueError):
        adapter.coerce_labels(bad)


# --- semantic validation is reused, not reimplemented ----------------------
@pytest.mark.parametrize("statement", [None, "", "   ", "\t\n"])
def test_unusable_statement_raises_invalid_point_error(statement):
    with pytest.raises(InvalidPointError) as e:
        adapter.prepare([{"statement": statement, "row_id": 4827}])
    assert "4827" in str(e.value), "the error must name the offending row"


def test_label_count_mismatch_raises_before_anything_else():
    with pytest.raises(LabelCountMismatch):
        adapter.prepare([{"statement": "a"}, {"statement": "b"}], labels=[1])


def test_prepare_returns_rows_and_binary_labels():
    rows, y = adapter.prepare(
        [{"statement": "a"}, {"statement": "b"}], labels=["true", "false"])
    assert [r.statement for r in rows] == ["a", "b"]
    assert y == [1, 0]


def test_empty_batch_is_not_an_error():
    """An empty batch is a defined no-op for a service, not an error."""
    rows, y = adapter.prepare([], labels=[])
    assert rows == [] and y == []


# --- ceilings --------------------------------------------------------------
def test_check_batch_names_the_limit_and_the_count():
    with pytest.raises(BatchTooLarge) as e:
        adapter.check_batch(501, 500, "points", "predict")
    assert "501" in str(e.value) and "500" in str(e.value)


def test_check_batch_allows_exactly_the_limit():
    adapter.check_batch(500, 500, "points", "predict")

"""Conversion between JSON tool arguments and truthclf.data.Row objects.

The truthclf package works with Row dataclasses and deliberately contains no
deserialisation layer; MCP tools receive JSON. This module is the single place
that bridges the two, so there is exactly one implementation to keep correct.

Input is checked in two passes:

  * the pydantic model below becomes each tool's published JSON schema and is
    enforced by the MCP framework before a handler body runs. It checks
    structure only -- key names, types, unknown fields.
  * truthclf.predictors.base.validate_points checks meaning: a null, empty or
    whitespace-only statement, and a labels array whose length does not match
    the points array.

The second pass is not redundant. A JSON schema can require that `statement` is
a string but not that "   " is unusable, and the failure it prevents is silent
rather than structural: prompt assembly falls back to the raw statement when the
cleaned one is empty, so a blank statement is rendered into the prompt and the
model returns a probability that looks exactly like a real one.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from truthclf.data import (LABELS_6, Row, binarize, clean_text,
                           normalized_statement_key)
from truthclf.predictors.base import validate_points

from .errors import BatchTooLarge

# A label on the wire is either the binary target or one of the six human
# labels. Declared as a Literal union so the permitted values appear in the
# published JSON schema instead of only in prose.
LabelValue = Literal[0, 1] | Literal[
    "true", "mostly-true", "half-true", "barely-true", "false", "extremely-false"]

Scheme = Literal["primary", "sensitivity"]

# Metadata fields in Row declaration order. Empty string is the correct default:
# it is what the CSV loader produces for an absent column, and prompt assembly
# renders an empty field as "[unknown]". Absent metadata is therefore a
# well-defined input, not an error; only an absent statement is an error.
METADATA_FIELDS = ("subjects", "speaker_name", "speaker_job", "speaker_state",
                   "speaker_affiliation", "statement_context")


class Point(BaseModel):
    """One statement with its attributes, excluding the label."""

    model_config = ConfigDict(extra="ignore")

    statement: str = Field(description="The claim being evaluated. Required.")
    row_id: int | None = Field(
        default=None,
        description="Optional. Defaults to this point's 0-based index in the "
                    "request. Echoed in every output so results can be joined back.")
    subjects: str = Field(default="", description="$-separated, e.g. 'state-finances$taxes'")
    speaker_name: str = ""
    speaker_job: str = ""
    speaker_state: str = ""
    speaker_affiliation: str = ""
    statement_context: str = ""


def to_row(point: Point | dict, index: int, label: str = "") -> Row:
    """Build a Row from a point, matching what the CSV loader produces.

    `label` defaults to empty because prediction and explanation inputs arrive
    without ground truth. That default is safe only because neither path calls
    Row.y(), which looks the label up in the binarisation table and would raise
    KeyError on an empty string. Ground truth reaches the metrics through the
    separate labels array (see coerce_labels). A caller that needs y() must pass
    a real label here.
    """
    d = point.model_dump() if isinstance(point, Point) else dict(point)
    statement = d.get("statement")
    # A null statement is passed through as an empty string rather than coerced
    # here, so that validate_points is the component that rejects it and names
    # the offending row.
    text = statement if isinstance(statement, str) else ("" if statement is None else str(statement))
    row_id = d.get("row_id")
    return Row(
        row_id=index if row_id is None else int(row_id),
        label=label,
        statement=text,
        **{f: (d.get(f) or "") for f in METADATA_FIELDS},
        # Both derived fields must be computed here, exactly as the CSV loader
        # computes them. Row defaults them to empty strings, and prompt assembly
        # uses `statement_clean or statement` -- so omitting statement_clean
        # sends the raw statement to the model instead of the cleaned one:
        # every probability shifts, nothing raises, and results stop matching
        # the recorded evaluation. norm_key is the grouping key behind
        # split membership and duplicate detection.
        statement_clean=clean_text(text),
        norm_key=normalized_statement_key(text),
    )


def to_rows(points, offset: int = 0) -> list[Row]:
    """Build Rows for a batch, assigning index-based row_ids where absent."""
    return [to_row(p, offset + i) for i, p in enumerate(points)]


def coerce_labels(labels, scheme: str = "primary") -> list[int] | None:
    """Convert wire labels to a list of ints in {0, 1}, or None.

    Accepts the binary target directly, or any of the six human labels, which
    are mapped through the package's own binarize() rather than a local copy of
    the table. The six-to-two mapping is a deliberate design choice with a
    documented sensitivity variant; a second copy here would be a second place
    for it to drift.
    """
    if labels is None:
        return None
    out = []
    for i, v in enumerate(labels):
        if isinstance(v, bool):
            # bool subclasses int, so accepting it would admit a type the
            # published schema does not advertise.
            raise ValueError(f"labels[{i}] is a bool; use 0/1 or a 6-way label string")
        if isinstance(v, int):
            if v not in (0, 1):
                raise ValueError(f"labels[{i}] = {v!r} is not in {{0, 1}}")
            out.append(v)
        elif isinstance(v, str):
            key = v.strip().lower()
            if key not in LABELS_6:
                raise ValueError(
                    f"labels[{i}] = {v!r} is not one of {LABELS_6} nor 0/1")
            out.append(binarize(key, scheme))
        else:
            raise ValueError(f"labels[{i}] has unsupported type {type(v).__name__}")
    return out


def prepare(points, labels=None, scheme: str = "primary"):
    """Build Rows, convert labels, then validate. Returns (rows, labels).

    Validation runs last and before the caller touches any client, which is
    where it belongs: it is the check that stops an unusable point from reaching
    the provider and returning as a plausible-looking number.
    """
    rows = to_rows(points)
    y = coerce_labels(labels, scheme)
    validate_points(rows, y)
    return rows, y


# ---------------------------------------------------------------------------
# Batch ceilings
# ---------------------------------------------------------------------------
# Enforced in the handlers rather than as a pydantic list length, so the failure
# is a BatchTooLarge naming both the limit and the actual count rather than a
# generic schema validation error. Each limit is also published in the relevant
# tool's field description so a client can avoid the round trip.
#
# The defaults below are derived from the client's concurrency and recorded
# per-call latency, not measured end to end through this transport. They are
# starting values, overridable per deployment.
def _limit(env: str, default: int) -> int:
    raw = os.environ.get(env)
    return default if raw is None else int(raw)


MAX_PREDICT_POINTS = _limit("TRUTHCLF_MCP_MAX_PREDICT_POINTS", 500)
MAX_EXPLAIN_POINTS = _limit("TRUTHCLF_MCP_MAX_EXPLAIN_POINTS", 50)
MAX_DATASET_ROWS = _limit("TRUTHCLF_MCP_MAX_DATASET_ROWS", 1000)
MAX_METRICS_LENGTH = _limit("TRUTHCLF_MCP_MAX_METRICS_LENGTH", 100_000)
MAX_RETRIEVAL_QUERIES = _limit("TRUTHCLF_MCP_MAX_RETRIEVAL_QUERIES", 500)


def check_batch(n: int, limit: int, unit: str, tool: str, note: str = "") -> None:
    """Refuse an oversized batch. Never truncates."""
    if n > limit:
        raise BatchTooLarge(
            f"{tool}: {n} {unit} exceeds the ceiling of {limit}. "
            f"{note}Split the request; this tool never truncates, because "
            "metrics over a silently shortened set answer a different question "
            "than the one asked.")

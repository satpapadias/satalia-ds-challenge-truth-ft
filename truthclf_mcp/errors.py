"""Error types for the MCP layer, and the single place library errors are mapped.

Errors that cross the MCP boundary carry their class name, because the class is
the contract: a caller can act on "InvalidPointError: point 4827 has no usable
statement" but not on a generic execution failure.

Nothing here is caught and swallowed. Every handler either returns a result or
raises.
"""

from __future__ import annotations

import functools

from mcp.server.mcpserver.exceptions import ToolError

from truthclf.evaluation import CalibratorModelMismatch
from truthclf.explain import DegradedPredictions
from truthclf.predictors.base import InvalidPointError, LabelCountMismatch


class BatchTooLarge(ValueError):
    """More items than a tool's ceiling. Names the limit and the actual count.

    Requests are refused rather than truncated: metrics computed over a silently
    shortened set answer a different question than the one the caller asked.
    """


class FineTunedRowNotCached(KeyError):
    """A row has no stored fine-tuned probability under source="cached".

    Never falls back to a live call or to the zero-shot predictor.
    """


class FineTunedModelNotServable(RuntimeError):
    """The fine-tuned adapter cannot be served by the provider.

    Together returns HTTP 400 with code `model_not_available` ("Unable to access
    non-serverless model") for this adapter. That is a capability statement, not
    a transient failure, and it is correctly excluded from the retry policy.
    """


class CounterfactualNotAvailable(RuntimeError):
    """Occlusion was requested against a predictor backed by stored probabilities.

    The fine-tuned probability store is keyed on row_id alone, and occlusion
    builds field-ablated copies of a row that keep the same row_id. Serving those
    from the store would return the base row's probability for every variant:
    all deltas 0.0, every point attributed to the statement, and a complete,
    plausible-looking driver distribution that is an artifact of the join key.
    A response cache cannot answer counterfactual queries.
    """


class ProviderCredentialMissing(RuntimeError):
    """A live provider call was requested with no API key configured."""


class RowNotInSplit(KeyError):
    """Requested row_ids are absent from the requested split. Never dropped silently."""


class LeakageAssertionFailed(RuntimeError):
    """A retrieval result contained a row outside the train split.

    Raised with no partial results. Train-only is the constraint the tool exists
    to honour, so a violation must not degrade into a warning.
    """


# Exceptions meaning "the caller sent something unusable" or "this operation
# cannot be honoured", as distinct from a defect in this layer. Their messages
# are already precise -- InvalidPointError names the offending row_id, the
# fine-tuned probability loader names the missing count and the first few ids --
# so they are surfaced verbatim rather than rewritten.
_PASS_THROUGH = (
    InvalidPointError,
    LabelCountMismatch,
    CalibratorModelMismatch,
    DegradedPredictions,
    BatchTooLarge,
    FineTunedRowNotCached,
    FineTunedModelNotServable,
    CounterfactualNotAvailable,
    ProviderCredentialMissing,
    RowNotInSplit,
    LeakageAssertionFailed,
)


def tool_errors(fn):
    """Map known caller-facing exceptions onto ToolError, class name preserved.

    Unrecognised exceptions are deliberately not caught: they propagate to the
    SDK, which reports them as an execution failure. A caller should be able to
    tell "you sent me a bad point" from "this server has a bug", and a blanket
    handler erases that distinction.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _PASS_THROUGH as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper

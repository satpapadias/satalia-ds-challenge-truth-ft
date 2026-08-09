"""Shared predictor interface.

Every predictor processes a batch of rows and returns a PredictionResult. When
ground-truth labels are supplied, the result also carries evaluation metrics, so
the zero-shot and fine-tuned predictors are fully interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .. import metrics as M


class InvalidPointError(ValueError):
    """A point is not usable as model input.

    Raised at the predictor boundary rather than deeper down, because the
    failure mode it prevents is silent: `build_user_prompt` uses
    `statement_clean or statement`, so a null statement renders the literal
    string "None" into the prompt and the model happily scores it. No error, no
    warning, and a probability that looks exactly like a real one.
    """


class LabelCountMismatch(ValueError):
    """len(labels) != len(points). Metrics computed from misaligned arrays are
    silently wrong, so this is checked before any model call."""


def validate_points(points, labels=None) -> None:
    """Reject inputs that would otherwise produce a plausible-looking answer."""
    if labels is not None and len(labels) != len(points):
        raise LabelCountMismatch(
            f"{len(labels)} labels for {len(points)} points")
    for i, p in enumerate(points):
        statement = getattr(p, "statement_clean", None) or getattr(p, "statement", None)
        if statement is None or not str(statement).strip():
            rid = getattr(p, "row_id", f"<index {i}>")
            raise InvalidPointError(
                f"point {rid} has no usable statement "
                f"(statement={getattr(p, 'statement', None)!r}, "
                f"statement_clean={getattr(p, 'statement_clean', None)!r})")


@dataclass
class PredictionResult:
    scores: list           # raw 0-100 truthfulness scores (post-fallback)
    probs: list            # scores mapped to [0, 1]
    preds: list            # binary predictions at `threshold`
    threshold: float
    parse_failures: int    # how many model outputs failed to parse a score
    n: int
    metrics: dict = field(default=None)


def compute_metrics(y_true, preds, probs) -> dict:
    """Standard metric bundle for a set of predictions with probabilities.

    Delegates to metrics.metric_bundle. This used to be a second, hand-listed
    copy of the same bundle that had already drifted — it was missing the
    per-class precision/recall keys — so a predictor's `.metrics` and the
    evaluation scripts' tables silently reported different things.
    """
    return M.metric_bundle(y_true, preds, probs)


class Predictor(ABC):
    @abstractmethod
    def predict(self, rows, labels=None) -> PredictionResult:
        """Predict over a batch of rows; include metrics if labels are given."""
        raise NotImplementedError

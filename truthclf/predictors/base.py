"""Shared predictor interface.

Every predictor processes a batch of rows and returns a PredictionResult. When
ground-truth labels are supplied, the result also carries evaluation metrics, so
the zero-shot and fine-tuned predictors are fully interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .. import metrics as M


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

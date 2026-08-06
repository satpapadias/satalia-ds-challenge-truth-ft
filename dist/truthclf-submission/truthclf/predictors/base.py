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
    """Standard metric bundle for a set of predictions with probabilities."""
    return {
        "accuracy": M.accuracy(y_true, preds),
        "balanced_accuracy": M.balanced_accuracy(y_true, preds),
        "macro_f1": M.macro_f1(y_true, preds),
        "roc_auc": M.roc_auc(y_true, probs),
        "pr_auc": M.pr_auc(y_true, probs),
        "brier": M.brier(y_true, probs),
        "ece": M.ece(y_true, probs),
    }


class Predictor(ABC):
    @abstractmethod
    def predict(self, rows, labels=None) -> PredictionResult:
        """Predict over a batch of rows; include metrics if labels are given."""
        raise NotImplementedError

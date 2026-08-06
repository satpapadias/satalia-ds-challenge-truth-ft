"""Selective prediction (abstention): accuracy as the model abstains on its
least-confident predictions.

Confidence is |p - threshold| (distance from the decision boundary). Rows are
ranked most-confident first; the coverage-vs-accuracy curve reports accuracy over
the most-confident fraction kept.
"""

from __future__ import annotations

import numpy as np


def coverage_accuracy_curve(probs, labels, threshold=0.5):
    """Return (coverage, accuracy) arrays, evaluated over the most-confident
    prefix of the data (coverage from 1/n .. 1.0)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    preds = (probs >= threshold).astype(int)
    conf = np.abs(probs - threshold)
    order = np.argsort(-conf)                 # most confident first
    correct = (preds[order] == labels[order]).astype(float)
    n = len(labels)
    coverage = np.arange(1, n + 1) / n
    accuracy = np.cumsum(correct) / np.arange(1, n + 1)
    return coverage, accuracy


def accuracy_at_coverage(probs, labels, coverage, threshold=0.5) -> float:
    """Accuracy over the most-confident `coverage` fraction of the data."""
    cov, acc = coverage_accuracy_curve(probs, labels, threshold)
    idx = min(len(acc) - 1, max(0, int(round(coverage * len(acc))) - 1))
    return float(acc[idx])

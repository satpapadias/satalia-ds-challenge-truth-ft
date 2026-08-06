"""Decision-threshold tuning on a validation split.

Two flavours:
  - tune_threshold: maximise balanced accuracy or macro-F1.
  - tune_cost_threshold: minimise expected cost where a false negative (passing a
    falsehood as True) is costlier than a false positive.

Tune on validation, apply the chosen threshold to test (tuning on test leaks).
"""

from __future__ import annotations

import numpy as np

from . import metrics as M

_GRID = np.linspace(0.01, 0.99, 197)        # 0.005 steps


def predict_at(probs, threshold) -> np.ndarray:
    return (np.asarray(probs, dtype=float) >= threshold).astype(int)


def tune_threshold(probs, labels, objective="balanced_accuracy", grid=_GRID):
    """Return (best_threshold, best_value) for the given objective."""
    fn = {"balanced_accuracy": M.balanced_accuracy, "macro_f1": M.macro_f1}[objective]
    best_thr, best_val = 0.5, -1.0
    for thr in grid:
        val = fn(labels, predict_at(probs, thr))
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val


def tune_cost_threshold(probs, labels, c_fn=2.0, c_fp=1.0, grid=_GRID):
    """Return (best_threshold, best_cost) minimising c_fn*FN + c_fp*FP."""
    best_thr, best_cost = 0.5, float("inf")
    for thr in grid:
        c = M.confusion(labels, predict_at(probs, thr))
        cost = c_fn * c["fn"] + c_fp * c["fp"]
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr, best_cost

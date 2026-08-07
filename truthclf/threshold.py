"""Decision-threshold tuning on a validation split.

Two flavours:
  - tune_threshold: maximise balanced accuracy or macro-F1.
  - tune_cost_threshold: minimise expected cost where a false negative (passing a
    falsehood as True) is costlier than a false positive.

Tune on validation, apply the chosen threshold to test (tuning on test leaks).

DECISION RULE: `score >= threshold` predicts class 1, everywhere in this module
and in every caller (see `predict_at`). The candidate set below is built from
sklearn.metrics.roc_curve, whose thresholds are defined for exactly that `>=`
convention. If the rule were ever changed to `>`, the candidates would be off by
one operating point on every tied score — which matters here, because this
dataset's scores are heavily tied (score-mode elicitation emits ~17 distinct
values).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

from . import metrics as M


def predict_at(probs, threshold) -> np.ndarray:
    """Predict class 1 where score >= threshold. See the module docstring."""
    return (np.asarray(probs, dtype=float) >= threshold).astype(int)


def candidate_thresholds(probs, labels) -> np.ndarray:
    """The achievable operating points for `probs`, ascending.

    These are the distinct score values, obtained from roc_curve. Scanning a
    fixed uniform grid instead (the previous approach: 197 points at 0.005
    spacing) is both wasteful and wrong here — with only ~17 distinct scores,
    most grid points are duplicates of each other, and a grid point that lands
    exactly ON a score value makes the selected threshold depend on whether the
    comparison is >= or >.

    Two roc_curve behaviours are handled explicitly:

    * `drop_intermediate=False` is REQUIRED. The default prunes thresholds that
      are not on the convex hull of the ROC curve, which is right for plotting
      and wrong here: a pruned threshold can still be the optimum for balanced
      accuracy, macro-F1 or an asymmetric cost, and dropping it silently biases
      the tuned threshold upward.
    * roc_curve prepends an artificial threshold above max(score) (np.inf in
      current sklearn, max+1 historically) so the curve starts at (0, 0). That
      point means "predict everything negative" and is not a real operating
      point, so it is dropped.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.min() == y.max():
        raise ValueError("threshold tuning needs both classes in `labels`")
    _, _, thr = roc_curve(y, probs, drop_intermediate=False)
    thr = np.asarray(thr[1:], dtype=float)          # drop the artificial +inf point
    assert np.all(np.isfinite(thr)), "roc_curve returned a non-finite threshold"
    lo, hi = float(np.min(probs)), float(np.max(probs))
    assert np.all((thr >= lo) & (thr <= hi)), (
        f"candidate thresholds must lie within [{lo}, {hi}], got "
        f"[{thr.min()}, {thr.max()}]")
    return np.sort(thr)                             # ascending: preserves tie-breaking


def tune_threshold(probs, labels, objective="balanced_accuracy", grid=None):
    """Return (best_threshold, best_value) for the given objective.

    Ties are broken toward the LOWEST threshold achieving the optimum (strict
    `>` on the objective while scanning ascending candidates) — unchanged from
    the previous implementation.
    """
    fn = {"balanced_accuracy": M.balanced_accuracy, "macro_f1": M.macro_f1}[objective]
    cands = candidate_thresholds(probs, labels) if grid is None else np.sort(np.asarray(grid))
    best_thr, best_val = 0.5, -1.0
    for thr in cands:
        val = fn(labels, predict_at(probs, thr))
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val


def tune_cost_threshold(probs, labels, c_fn=2.0, c_fp=1.0, grid=None):
    """Return (best_threshold, best_cost) minimising c_fn*FN + c_fp*FP.

    Ties break toward the lowest threshold, as in tune_threshold.
    """
    cands = candidate_thresholds(probs, labels) if grid is None else np.sort(np.asarray(grid))
    best_thr, best_cost = 0.5, float("inf")
    for thr in cands:
        c = M.confusion(labels, predict_at(probs, thr))
        cost = c_fn * c["fn"] + c_fp * c["fp"]
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr, best_cost

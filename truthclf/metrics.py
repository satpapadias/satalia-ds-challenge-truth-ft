"""Evaluation metrics for binary truthfulness classification.

Thin, explicit wrappers over reference implementations — scikit-learn for the
classification/ranking/calibration metrics, statsmodels for McNemar — plus the
few things those libraries do not provide:

  - ece                    (no scikit-learn equivalent)
  - reliability_curve      (calibration_curve drops empty bins and returns no counts)
  - bootstrap_bundle       (one shared resample feeding every metric, for cost)

The wrappers exist so call sites keep a stable signature and so every
edge-case choice (zero_division, labels, the degenerate-input convention) is
made once, visibly, here rather than being re-decided at each call.

Conventions: y_true is array-like of {0,1}; y_pred is {0,1}; y_score/y_prob is
a real-valued / [0,1] confidence that the label is 1 (True).

Degenerate-input convention: metrics that are mathematically undefined return
NaN rather than a plausible-looking number. bootstrap_ci drops NaN resamples and
warns with a count, so a degenerate draw narrows the interval visibly instead of
silently.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn import metrics as skm
from statsmodels.stats.contingency_tables import mcnemar as _sm_mcnemar

# Both classes are always named explicitly so a resample (or a degenerate split)
# that happens to contain only one class still yields a 2x2 shape.
LABELS = [0, 1]


def _arr(x):
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred) -> dict:
    # labels=LABELS pins the matrix to 2x2 even when one class is absent, which
    # ravel() would otherwise silently mis-unpack.
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def accuracy(y_true, y_pred) -> float:
    return float(skm.accuracy_score(y_true, y_pred))


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of per-class recall (TPR and TNR).

    Note a deliberate difference from the previous hand-rolled version: if a
    class is absent from y_true its recall is undefined, and sklearn averages
    over the classes that are present (the old code used an epsilon denominator
    and folded in a 0.0). Both classes are present in every split we report, so
    no reported number changes; the sklearn behaviour is the correct one.
    """
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of F1 for class 0 and class 1.

    zero_division=0.0: if a class is never predicted AND never present, its F1 is
    undefined and counted as 0, which keeps the previous (deliberately
    unflattering) semantics. zero_division=np.nan would instead drop the class
    from the mean and report 1.0 for a run that never tested one class at all.
    """
    return float(skm.f1_score(y_true, y_pred, labels=LABELS, average="macro",
                              zero_division=0.0))


def precision_recall(y_true, y_pred) -> dict:
    """Per-class precision and recall: {cls: (precision, recall)} for cls in 0,1.

    zero_division=np.nan here, unlike macro_f1: these values are reported
    individually, and an undefined per-class precision should read as undefined
    rather than as a score of 0.
    """
    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=np.nan)
    return {cls: (float(prec[i]), float(rec[i])) for i, cls in enumerate(LABELS)}


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def roc_auc(y_true, y_score) -> float:
    """ROC AUC, tie-aware. NaN when only one class is present (undefined)."""
    yt = _arr(y_true).astype(int)
    if yt.size == 0 or yt.min() == yt.max():
        # Checked here rather than letting sklearn warn-and-return-NaN, so a
        # bootstrap over degenerate draws does not emit thousands of warnings.
        return float("nan")
    return float(skm.roc_auc_score(yt, _arr(y_score)))


def pr_auc(y_true, y_score) -> float:
    """Average precision (area under the precision-recall curve).

    Delegates to sklearn, which steps over DISTINCT score thresholds. The
    previous hand-rolled version accumulated precision per SAMPLE, so inside a
    group of tied scores it credited operating points the classifier cannot
    actually realise — making the result depend on row order. With every score
    tied it returned 0.417 / 0.833 / 1.000 for the same labels in different
    orders, where the correct answer is the positive prevalence in all three.

    NaN when there are no positives (undefined); sklearn returns 0.0 there,
    which is indistinguishable from a genuinely terrible ranking.
    """
    yt = _arr(y_true).astype(int)
    if not yt.any():
        return float("nan")
    return float(skm.average_precision_score(yt, _arr(y_score)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def brier(y_true, y_prob) -> float:
    """Mean squared error between probability and outcome."""
    return float(skm.brier_score_loss(y_true, _arr(y_prob), pos_label=1))


def reliability_curve(y_true, y_prob, n_bins=10):
    """Equal-width bins over [0,1]. Returns per-bin dict arrays:
    mean_pred, frac_pos, count (empty bins reported with count=0).

    Kept hand-rolled: sklearn.calibration.calibration_curve silently DROPS empty
    bins and returns no per-bin counts, so a caller cannot tell a well-populated
    bin from one holding a single point, nor see which bins were empty. Both are
    needed by viz.reliability_plot and by the ECE diagnostics. Pinned against
    calibration_curve on the non-empty bins in tests/test_metrics.py.
    """
    yt, p = _arr(y_true), _arr(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, count = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        k = int(np.sum(mask))
        count.append(k)
        mean_pred.append(float(np.mean(p[mask])) if k else float("nan"))
        frac_pos.append(float(np.mean(yt[mask])) if k else float("nan"))
    return {"mean_pred": mean_pred, "frac_pos": frac_pos, "count": count,
            "edges": edges.tolist()}


def ece(y_true, y_prob, n_bins=10) -> float:
    """Expected Calibration Error using confidence of the predicted class.

    Kept hand-rolled: scikit-learn has no ECE, and torchmetrics/netcal are not
    worth a dependency for 15 lines. Pinned in tests against an analytically
    known case and against calibration_curve's binning.

    KNOWN RESOLUTION LIMIT: confidence-of-predicted-class lives in [0.5, 1] for
    binary classification, but the bins are equal-width over [0, 1]. The lower
    half is therefore structurally empty and the effective resolution is
    n_bins/2, not n_bins — a caller asking for 10 bins gets 5. Use
    `ece_bin_report` to see the occupancy directly.
    """
    return _ece_bins(y_true, y_prob, n_bins)["ece"]


def ece_bin_report(y_true, y_prob, n_bins=10) -> dict:
    """ECE plus the per-bin occupancy it was computed from.

    Exists so the structural-empty-bin limit documented on `ece` is visible to
    the caller instead of being folded into a bare float: `n_bins_occupied` vs
    `n_bins` shows the effective resolution directly.
    """
    return _ece_bins(y_true, y_prob, n_bins)


def _ece_bins(y_true, y_prob, n_bins) -> dict:
    yt, p = _arr(y_true), _arr(y_prob)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)        # confidence in predicted class
    correct = (pred == yt).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(yt)
    total = 0.0
    counts, gaps = [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        k = int(np.sum(mask))
        counts.append(k)
        if k:
            gap = abs(np.mean(correct[mask]) - np.mean(conf[mask]))
            gaps.append(float(gap))
            total += (k / n) * gap
        else:
            gaps.append(float("nan"))
    return {"ece": float(total), "n_bins": n_bins,
            "n_bins_occupied": int(sum(1 for c in counts if c)),
            "count": counts, "gap": gaps, "edges": edges.tolist()}


# ---------------------------------------------------------------------------
# Uncertainty & paired testing
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn, *arrays, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a metric over paired arrays.

    Returns (point_estimate, lo, hi). `metric_fn` takes the arrays in order.

    A resample that yields NaN (e.g. a draw with one class absent, so AUC is
    undefined) is dropped and counted; if that happens the interval is computed
    over fewer than n_boot draws and a warning names the count, because a
    quietly narrowed interval looks exactly like a confident one. Exceptions are
    NOT caught: `except Exception: continue` here silently discarded genuine
    bugs in metric_fn as if they were degenerate draws.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(float(metric_fn(*[a[idx] for a in arrays])))
    kept = np.array([s for s in stats if not math.isnan(s)])
    n_dropped = len(stats) - len(kept)
    if n_dropped:
        warnings.warn(
            f"bootstrap_ci: {n_dropped}/{n_boot} resamples returned NaN and were "
            f"dropped; the interval is computed over {len(kept)} draws.",
            RuntimeWarning, stacklevel=2,
        )
    if len(kept) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(kept, 100 * alpha / 2))
    hi = float(np.percentile(kept, 100 * (1 - alpha / 2)))
    return point, lo, hi


def metric_bundle(y_true, preds, probs) -> dict:
    """Flat dict of all scalar metrics for a set of predictions."""
    pr = precision_recall(y_true, preds)
    return {
        "accuracy": accuracy(y_true, preds),
        "balanced_accuracy": balanced_accuracy(y_true, preds),
        "macro_f1": macro_f1(y_true, preds),
        "precision_True": pr[1][0], "recall_True": pr[1][1],
        "precision_False": pr[0][0], "recall_False": pr[0][1],
        "roc_auc": roc_auc(y_true, probs),
        "pr_auc": pr_auc(y_true, probs),
        "brier": brier(y_true, probs),
        "ece": ece(y_true, probs),
    }


def bootstrap_bundle(y_true, preds, probs, n_boot=1000, seed=0, alpha=0.05) -> dict:
    """Percentile-bootstrap CIs for every metric in metric_bundle, using one
    shared resample per iteration. Returns {metric: (point, lo, hi)}.

    Kept hand-rolled purely for COST: one pass computes all 11 metrics per
    resample, where scipy.stats.bootstrap would need 11 independent runs of
    n_boot draws each. Sharing the draw is not a statistical requirement — each
    metric's marginal CI is identically distributed either way, and we never do
    joint inference across metrics.
    """
    yt, pp, pb = _arr(y_true), _arr(preds), _arr(probs)
    n = len(yt)
    rng = np.random.default_rng(seed)
    point = metric_bundle(yt, pp, pb)
    samples = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = metric_bundle(yt[idx], pp[idx], pb[idx])
        for k, v in b.items():
            samples[k].append(v)
    out = {}
    for k, pt in point.items():
        arr = np.array([s for s in samples[k] if not math.isnan(s)])
        if len(arr) == 0:
            out[k] = (pt, float("nan"), float("nan"))
            continue
        if len(arr) < n_boot:
            warnings.warn(
                f"bootstrap_bundle: {n_boot - len(arr)}/{n_boot} resamples were NaN "
                f"for {k!r} and were dropped.", RuntimeWarning, stacklevel=2)
        out[k] = (pt, float(np.percentile(arr, 100 * alpha / 2)),
                  float(np.percentile(arr, 100 * (1 - alpha / 2))))
    return out


def paired_accuracy_diff(y_true, pred_a, pred_b, n_boot=2000, seed=0, alpha=0.05):
    """Bootstrap CI for accuracy(A) - accuracy(B) on the SAME test set.

    Returns (point, lo, hi). Resampling is paired: one index draw is applied to
    both predictors, so the CI is on the difference rather than on two
    independent accuracies. Lives here rather than in an evaluation script,
    which previously carried its own copy of this loop.
    """
    yt, a, b = _arr(y_true), _arr(pred_a), _arr(pred_b)
    n = len(yt)
    rng = np.random.default_rng(seed)
    point = float(np.mean(a == yt) - np.mean(b == yt))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = np.mean(a[idx] == yt[idx]) - np.mean(b[idx] == yt[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """McNemar's test for two classifiers on the SAME test set.

    b = #(A correct, B wrong); c = #(A wrong, B correct).

    `p_value` is the two-sided EXACT binomial test over the n=b+c discordant
    pairs — the reported figure, and the right choice at any n. `chi2` and
    `p_value_chi2` are the continuity-corrected chi-square approximation,
    reported alongside for reference. The previous hand-rolled version returned
    the exact p next to a chi2 from a different test with no indication they were
    not a matched pair, and its chi2 formula was unclamped, yielding 1/n instead
    of 0 when b == c.
    """
    yt = _arr(y_true).astype(int)
    a_ok = (_arr(pred_a).astype(int) == yt)
    b_ok = (_arr(pred_b).astype(int) == yt)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0,
                "p_value_chi2": 1.0}
    # Only the off-diagonal counts enter the test; the diagonal is filled for shape.
    table = [[int(np.sum(a_ok & b_ok)), b], [c, int(np.sum(~a_ok & ~b_ok))]]
    exact = _sm_mcnemar(table, exact=True)
    approx = _sm_mcnemar(table, exact=False, correction=True)
    return {"b": b, "c": c, "n_discordant": n,
            "chi2": float(approx.statistic), "p_value": float(exact.pvalue),
            "p_value_chi2": float(approx.pvalue)}

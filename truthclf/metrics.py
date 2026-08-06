"""Evaluation metrics for binary truthfulness classification.

Pure Python + numpy (no scipy/sklearn dependency):
  - balanced_accuracy, macro_f1            (robust to class imbalance)
  - roc_auc, pr_auc (average precision)
  - brier, ece, reliability_curve          (calibration)
  - bootstrap_ci                           (CIs on any metric)
  - mcnemar                                (paired model comparison, exact)

Conventions: y_true is array-like of {0,1}; y_pred is {0,1}; y_score/y_prob is
a real-valued / [0,1] confidence that the label is 1 (True).
"""

from __future__ import annotations

import math
import numpy as np

EPS = 1e-12


def _arr(x):
    return np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred) -> dict:
    yt, yp = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(yt == yp))


def balanced_accuracy(y_true, y_pred) -> float:
    """Mean of per-class recall (TPR and TNR)."""
    c = confusion(y_true, y_pred)
    tpr = c["tp"] / (c["tp"] + c["fn"] + EPS)
    tnr = c["tn"] / (c["tn"] + c["fp"] + EPS)
    return float((tpr + tnr) / 2.0)


def _f1_for(positive_label, y_true, y_pred) -> float:
    yt, yp = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    tp = np.sum((yt == positive_label) & (yp == positive_label))
    fp = np.sum((yt != positive_label) & (yp == positive_label))
    fn = np.sum((yt == positive_label) & (yp != positive_label))
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return float(2 * prec * rec / (prec + rec + EPS))


def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of F1 for class 0 and class 1."""
    return float((_f1_for(0, y_true, y_pred) + _f1_for(1, y_true, y_pred)) / 2.0)


def precision_recall(y_true, y_pred) -> dict:
    """Per-class precision and recall: {cls: (precision, recall)} for cls in 0,1."""
    yt, yp = _arr(y_true).astype(int), _arr(y_pred).astype(int)
    out = {}
    for cls in (0, 1):
        tp = np.sum((yt == cls) & (yp == cls))
        fp = np.sum((yt != cls) & (yp == cls))
        fn = np.sum((yt == cls) & (yp != cls))
        out[cls] = (float(tp / (tp + fp + EPS)), float(tp / (tp + fn + EPS)))
    return out


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def roc_auc(y_true, y_score) -> float:
    """AUC via the Mann-Whitney U statistic with tie-aware average ranks."""
    yt = _arr(y_true).astype(int)
    s = _arr(y_score)
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s_sorted):                      # average ranks within ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0            # 1-based
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    sum_ranks_pos = float(np.sum(ranks[yt == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y_true, y_score) -> float:
    """Average precision: sum over recall increments of precision (sklearn AP)."""
    yt = _arr(y_true).astype(int)
    s = _arr(y_score)
    n_pos = int(np.sum(yt == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")      # descending score
    yt = yt[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / (tp + fp + EPS)
    recall = tp / n_pos
    ap, prev_recall = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def brier(y_true, y_prob) -> float:
    """Mean squared error between probability and outcome."""
    yt, p = _arr(y_true), _arr(y_prob)
    return float(np.mean((p - yt) ** 2))


def reliability_curve(y_true, y_prob, n_bins=10):
    """Equal-width bins over [0,1]. Returns per-bin dict arrays:
    mean_pred, frac_pos, count (empty bins reported with count=0)."""
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
    """Expected Calibration Error using confidence of the predicted class."""
    yt, p = _arr(y_true), _arr(y_prob)
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)        # confidence in predicted class
    correct = (pred == yt).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(yt)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf >= lo) & (conf < hi) if b < n_bins - 1 else (conf >= lo) & (conf <= hi)
        k = int(np.sum(mask))
        if k:
            total += (k / n) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(total)


# ---------------------------------------------------------------------------
# Uncertainty & paired testing
# ---------------------------------------------------------------------------
def bootstrap_ci(metric_fn, *arrays, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a metric over paired arrays.
    Returns (point_estimate, lo, hi). `metric_fn` takes the arrays in order."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stats.append(float(metric_fn(*[a[idx] for a in arrays])))
        except Exception:
            continue
    stats = np.array([s for s in stats if not math.isnan(s)])
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
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
    shared resample per iteration. Returns {metric: (point, lo, hi)}."""
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
        else:
            out[k] = (pt, float(np.percentile(arr, 100 * alpha / 2)),
                      float(np.percentile(arr, 100 * (1 - alpha / 2))))
    return out


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar's test for two classifiers on the SAME test set.

    b = #(A correct, B wrong); c = #(A wrong, B correct). Two-sided exact
    binomial p-value over the n=b+c discordant pairs (no scipy needed).
    Also returns the chi-square statistic with continuity correction.
    """
    yt = _arr(y_true).astype(int)
    a_ok = (_arr(pred_a).astype(int) == yt)
    b_ok = (_arr(pred_b).astype(int) == yt)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2.0 * tail)
    chi2 = (abs(b - c) - 1) ** 2 / n              # continuity-corrected
    return {"b": b, "c": c, "n_discordant": n, "chi2": float(chi2),
            "p_value": float(p_value)}

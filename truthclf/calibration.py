"""Post-hoc probability calibration for binary predictions.

The zero-shot probability is score/100, which is uncalibrated. We fit a
calibrator on the validation split only and apply it to the test split.

Two calibrators, both operating on the logit of the raw probability:
  - temperature scaling: p_cal = sigmoid(logit(p) / T)        (1 parameter)
  - Platt scaling:        p_cal = sigmoid(A * logit(p) + B)    (2 parameters)

fit_best picks whichever gives lower validation NLL. Pure numpy.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _nll(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(probs, labels):
    """1-D search for the temperature T minimising validation NLL."""
    z = _logit(probs)
    y = np.asarray(labels, dtype=float)
    best_T, best = 1.0, float("inf")
    for T in np.linspace(0.05, 10.0, 400):
        nll = _nll(_sigmoid(z / T), y)
        if nll < best:
            best, best_T = nll, T
    return {"method": "temperature", "T": float(best_T)}


def fit_platt(probs, labels, iters=2000, lr=0.5):
    """Gradient-descent fit of A, B for Platt scaling (logistic on the logit)."""
    z = _logit(probs)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = _sigmoid(A * z + B)
        g = p - y
        gA = float(np.mean(g * z))
        gB = float(np.mean(g))
        A -= lr * gA
        B -= lr * gB
    return {"method": "platt", "A": float(A), "B": float(B)}


def apply(probs, params):
    z = _logit(probs)
    if params["method"] == "temperature":
        return _sigmoid(z / params["T"])
    if params["method"] == "platt":
        return _sigmoid(params["A"] * z + params["B"])
    raise ValueError(f"unknown calibrator: {params.get('method')!r}")


def fit_best(probs, labels):
    """Fit both calibrators on (probs, labels); return the lower-val-NLL one."""
    y = np.asarray(labels, dtype=float)
    cands = [fit_temperature(probs, labels), fit_platt(probs, labels)]
    scored = [(p, _nll(apply(probs, p), y)) for p in cands]
    scored.sort(key=lambda t: t[1])
    best = scored[0][0]
    best["val_nll"] = scored[0][1]
    return best

"""Post-hoc probability calibration for binary predictions.

The zero-shot probability is score/100 (or a logprob softmax), which is
uncalibrated. We fit a calibrator on the validation split only and apply it to
the test split.

Two calibrators, both operating on the logit of the raw probability:
  - temperature scaling: p_cal = sigmoid(logit(p) / T)        (1 parameter)
  - Platt scaling:        p_cal = sigmoid(A * logit(p) + B)    (2 parameters)

fit_best picks whichever gives lower validation NLL.

Both fits are delegated to reference optimisers — scipy for the 1-D bounded
search, scikit-learn's logistic regression for Platt — rather than a hand-rolled
grid and a fixed-step gradient loop.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

EPS = 1e-6

# Temperature is searched on this interval. Bounded rather than unconstrained so
# a degenerate validation set cannot drive T to 0 or infinity.
T_BOUNDS = (0.05, 10.0)


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
    """Fit the temperature T minimising validation NLL.

    Uses scipy's bounded Brent search. The previous version scanned 400 points
    of linspace(0.05, 10.0), quantising T to steps of ~0.025 and reporting a
    grid point rather than the optimum.
    """
    z = _logit(probs)
    y = np.asarray(labels, dtype=float)
    res = minimize_scalar(lambda T: _nll(_sigmoid(z / T), y),
                          bounds=T_BOUNDS, method="bounded")
    if not res.success:
        raise RuntimeError(f"temperature fit did not converge: {res.message}")
    return {"method": "temperature", "T": float(res.x)}


def fit_platt(probs, labels):
    """Fit A, B for Platt scaling: a logistic regression on the logit.

    Turning regularisation OFF is essential and deliberate. Platt scaling is a
    maximum-likelihood fit of two parameters; scikit-learn's LogisticRegression
    defaults to L2 with C=1.0, which would shrink A toward 0 and quietly flatten
    the calibration curve. A regularised fit here looks completely correct and is
    a worse bug than the fixed-step gradient descent it replaces, which at least
    failed visibly by not converging.

    C=np.inf rather than penalty=None: `penalty` is deprecated in scikit-learn
    1.8 and removed in 1.10, and C=np.inf is the documented replacement for "no
    regularisation". Pinned by tests/test_calibration.py, which recovers known
    (A, B) from synthetic data and would fail if the fit were shrunk.
    """
    z = _logit(probs).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return {"method": "platt", "A": float(lr.coef_[0][0]), "B": float(lr.intercept_[0])}


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

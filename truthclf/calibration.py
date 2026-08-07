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


def nll_difference_ci(probs, labels, n_boot=1000, seed=0, alpha=0.05):
    """Paired bootstrap CI for (temperature NLL - Platt NLL) on the fitting set.

    Positive means Platt fits better. BOTH calibrators are REFITTED on each
    resample and scored on that same resample, which mirrors exactly what
    fit_best does — the question is "would this selection flip on another
    validation draw of this size", not "how noisy is a fixed pair of fits".

    Returns (point, lo, hi).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    rng = np.random.default_rng(seed)

    def diff(p, y):
        yf = y.astype(float)
        return (_nll(apply(p, fit_temperature(p, y)), yf)
                - _nll(apply(p, fit_platt(p, y)), yf))

    point = diff(probs, labels)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]
        if y.min() == y.max():          # Platt is undefined on one class
            continue
        draws.append(diff(probs[idx], y))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def fit_best(probs, labels, n_boot=1000, seed=0, alpha=0.05):
    """Fit both calibrators; return Platt ONLY if it beats temperature by a
    margin that survives a paired bootstrap of the validation NLL difference.

    PARSIMONY RULE: temperature scaling has one parameter, Platt has two. When
    the CI for (temperature NLL - Platt NLL) includes zero, the data does not
    distinguish them and the simpler model is kept. The rule is stated in terms
    of parameter count, decided before looking at which calibrator it favours.

    Motivation: raw NLL comparison selected Platt on a 0.3% margin on one run
    and paid 0.016 test ECE for it. Selecting on an unquantified margin is how
    a calibration choice becomes noise-fitting.

    The returned dict carries `val_nll`, `nll_diff_ci` and `selected_by`
    ("margin" when Platt won outright, "parsimony" when the tie-break applied).
    """
    y = np.asarray(labels, dtype=float)
    temp = fit_temperature(probs, labels)
    platt = fit_platt(probs, labels)
    point, lo, hi = nll_difference_ci(probs, labels, n_boot=n_boot, seed=seed, alpha=alpha)

    platt_wins = lo > 0.0                 # CI strictly above zero
    best = platt if platt_wins else temp
    best["val_nll"] = _nll(apply(probs, best), y)
    best["nll_diff_ci"] = (point, lo, hi)
    best["selected_by"] = "margin" if platt_wins else "parsimony"
    return best

"""Plotting helpers (display only) built on truthclf.metrics.

matplotlib is imported lazily so the rest of the package has no plotting
dependency. Functions take a ZeroShotRun (or probs+labels) and return a
matplotlib Axes for display in a notebook.
"""

from __future__ import annotations

from . import metrics as M


def reliability_plot(run_or_probs, labels=None, n_bins: int = 10, ax=None):
    """Reliability (calibration) curve: observed positive rate vs mean predicted
    probability per bin, with the diagonal as perfect calibration."""
    import matplotlib.pyplot as plt

    if labels is None:                       # a ZeroShotRun was passed
        probs, labels = run_or_probs.probs, run_or_probs.labels
        title = f"{run_or_probs.model.split('/')[-1]} [{run_or_probs.variant}]"
    else:
        probs, title = run_or_probs, "reliability"

    rc = M.reliability_curve(labels, probs, n_bins=n_bins)
    xs, ys, counts = rc["mean_pred"], rc["frac_pos"], rc["count"]
    pts = [(x, y, c) for x, y, c in zip(xs, ys, counts)
           if c > 0 and x == x and y == y]   # drop empty/NaN bins

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label="model")
    ax.set_xlabel("mean predicted P(True)")
    ax.set_ylabel("observed fraction True")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability — {title}")
    ax.legend(loc="upper left")
    ece = M.ece(labels, probs, n_bins=n_bins)
    brier = M.brier(labels, probs)
    ax.text(0.98, 0.02, f"ECE={ece:.3f}\nBrier={brier:.3f}",
            ha="right", va="bottom", transform=ax.transAxes)
    return ax


def reliability_before_after(probs_raw, probs_cal, labels, n_bins=10):
    """Two reliability diagrams side by side: raw vs calibrated probabilities."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    reliability_plot(probs_raw, labels, n_bins=n_bins, ax=axes[0])
    axes[0].set_title("Reliability — raw score/100")
    reliability_plot(probs_cal, labels, n_bins=n_bins, ax=axes[1])
    axes[1].set_title("Reliability — calibrated")
    fig.tight_layout()
    return fig


def coverage_plot(probs, labels, threshold=0.5, ax=None, label="model"):
    """Coverage-vs-accuracy curve for selective prediction (abstain on the
    least-confident rows)."""
    import matplotlib.pyplot as plt
    from . import selective

    cov, acc = selective.coverage_accuracy_curve(probs, labels, threshold)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov, acc, label=label)
    ax.axhline(acc[-1], ls="--", color="gray", label="full-coverage acc")
    ax.set_xlabel("coverage (fraction predicted)")
    ax.set_ylabel("accuracy on covered rows")
    ax.set_xlim(0, 1)
    ax.set_title("Selective prediction — coverage vs accuracy")
    ax.legend(loc="lower left")
    return ax

"""The calibrated-evaluation pipeline: the single definition of "our number".

Every reported figure in the adopted record comes from this sequence:

    1. fit a calibrator on the VALIDATION probabilities        (calibration.fit_best)
    2. apply it to both validation and test probabilities      (calibration.apply)
    3. tune the decision threshold on the CALIBRATED validation
       probabilities                                            (threshold.tune_threshold)
    4. predict on test with `calibrated >= threshold` and score it

Order matters and is easy to get subtly wrong. Tuning the threshold on RAW
validation probabilities and then applying it to CALIBRATED test probabilities
compares two different scales; so does calibrating after tuning. This module
exists because that sequence had been written out four separate times — three
identical copies plus one divergent one — and the calibrator-selection fix had to
be applied to each by hand.

Nothing here decides policy: the calibrator choice lives in calibration.fit_best
(validation NLL with a parsimony margin rule) and the threshold objective is the
caller's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import calibration, metrics as M, threshold as T


@dataclass
class CalibratedEvaluation:
    """Result of the calibrated pipeline on one test split."""

    calibrator: dict                     # full fit_best output (method + params)
    threshold: float
    preds: list
    probs_calibrated: list               # test probabilities after calibration
    probs_raw: list                      # test probabilities before calibration
    metrics: dict                        # metric_bundle on the calibrated probs
    val_probs_calibrated: list = field(default_factory=list)

    @property
    def method(self) -> str:
        return self.calibrator["method"]

    @property
    def selected_by(self) -> str:
        """"margin" if the calibrator won outright, "parsimony" on a tie-break."""
        return self.calibrator.get("selected_by", "unknown")


def calibrated_evaluation(val_probs, val_labels, test_probs, test_labels,
                          objective="balanced_accuracy", n_boot=1000, seed=0):
    """Fit on validation, report on test. See the module docstring for the order.

    `val_probs` / `test_probs` are raw model probabilities; labels are {0,1}.
    Returns a CalibratedEvaluation. The test split is never used for fitting or
    tuning — passing test labels here only scores the result.
    """
    cal = calibration.fit_best(val_probs, val_labels, n_boot=n_boot, seed=seed)
    val_cal = list(calibration.apply(val_probs, cal))
    test_cal = list(calibration.apply(test_probs, cal))
    thr, _ = T.tune_threshold(val_cal, val_labels, objective)
    preds = T.predict_at(test_cal, thr)
    return CalibratedEvaluation(
        calibrator=cal, threshold=float(thr), preds=preds.tolist(),
        probs_calibrated=[float(p) for p in test_cal],
        probs_raw=[float(p) for p in test_probs],
        metrics=M.metric_bundle(test_labels, preds, test_cal),
        val_probs_calibrated=[float(p) for p in val_cal],
    )

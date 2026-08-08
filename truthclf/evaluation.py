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


# ---------------------------------------------------------------------------
# Decision artifact: the fitted calibrator + tuned threshold, as a shippable file
# ---------------------------------------------------------------------------
ARTIFACT_SCHEMA = 2


class CalibratorModelMismatch(ValueError):
    """Raised when an artifact is applied to a model it was not fitted for.

    A calibrator maps one model's probability scale onto calibrated
    probabilities. Applying model A's mapping to model B's output is silently
    wrong — the numbers still look like probabilities — so this is a hard error,
    never a warning.
    """


@dataclass
class DecisionArtifact:
    """Everything needed to turn raw model probabilities into decisions.

    Without this, `predict()` returns RAW probabilities thresholded at 0.5, and
    reproducing the reported behaviour means refitting on the validation split
    at start-up. Persisting it makes the calibrator shippable: a container loads
    the file and applies it.
    """

    model: str                      # the model id this calibrator was fitted for
    calibrator: dict                # fit_best output: method + T or (A, B)
    threshold: float
    objective: str
    fitted_on: str                  # human-readable description of the fit split
    n_val: int
    candidate_val_nll: dict         # {"temperature": ..., "platt": ...}
    nll_diff_ci: list               # (point, lo, hi) for temperature - platt
    selected_by: str                # "margin" | "parsimony"
    schema: int = ARTIFACT_SCHEMA

    # No timestamp is stored. A wall-clock field makes every no-op regeneration
    # produce a different file, so `git diff` stops distinguishing "the numbers
    # changed" from "the script ran again". The write time is logged instead.

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["calibrator"] = dict(self.calibrator)
        return d

    def save(self, path: str) -> str:
        """Write the artifact. Deterministic: identical inputs -> identical bytes."""
        import datetime
        import json
        import logging
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        logging.getLogger(__name__).info(
            "wrote calibrator artifact %s for %s at %s", path, self.model,
            datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
        return path

    @classmethod
    def load(cls, path: str) -> "DecisionArtifact":
        import json
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(
                f"{path}: artifact schema {d.get('schema')} != expected "
                f"{ARTIFACT_SCHEMA}; refit rather than reinterpreting it "
                "(schema 2 dropped the created_at field)")
        return cls(**d)

    def check_model(self, model: str) -> None:
        """Hard-fail if this artifact was fitted for a different model."""
        if model != self.model:
            raise CalibratorModelMismatch(
                f"calibrator was fitted for {self.model!r} but is being applied to "
                f"{model!r}. A calibrator is specific to the probability scale of "
                f"the model that produced it; refit for this model instead.")

    def apply(self, probs):
        """Raw probabilities -> calibrated probabilities."""
        return list(calibration.apply(probs, self.calibrator))

    def decide(self, probs):
        """Raw probabilities -> (calibrated probabilities, binary predictions)."""
        cal = self.apply(probs)
        return cal, T.predict_at(cal, self.threshold).tolist()


def build_artifact(evaluation: CalibratedEvaluation, model: str, fitted_on: str,
                   n_val: int, val_probs=None, val_labels=None,
                   objective="balanced_accuracy") -> DecisionArtifact:
    """Package a CalibratedEvaluation into a shippable DecisionArtifact."""
    cal = evaluation.calibrator
    cands = {}
    if val_probs is not None and val_labels is not None:
        y = np.asarray(val_labels, dtype=float)
        for fit, name in ((calibration.fit_temperature, "temperature"),
                          (calibration.fit_platt, "platt")):
            params = fit(val_probs, val_labels)
            cands[name] = calibration._nll(calibration.apply(val_probs, params), y)
    return DecisionArtifact(
        model=model,
        calibrator={k: v for k, v in cal.items() if k != "nll_diff_ci"},
        threshold=float(evaluation.threshold),
        objective=objective,
        fitted_on=fitted_on,
        n_val=int(n_val),
        candidate_val_nll=cands,
        nll_diff_ci=[float(x) for x in cal.get("nll_diff_ci", ())],
        selected_by=cal.get("selected_by", "unknown"),
    )

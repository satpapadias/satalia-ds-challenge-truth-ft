"""Reconciling two predictors into one verdict.

Pure functions, no I/O, so the decision rule can be tested without agents.

The rule is log-odds pooling of the two calibrated probabilities. Averaging the
probabilities directly would be the obvious alternative and is wrong for
calibrated outputs: probabilities live on a bounded scale where the difference
between 0.90 and 0.95 represents far more evidence than the one between 0.50 and
0.55, so a linear average distorts the comparison. Log-odds is the scale the
calibrators were fitted on, and the scale on which the two are comparable.

The weights sum to one, which makes this an interpolation rather than an
accumulation: two predictors that agree do not produce a more confident pool
than either alone. That is deliberate. Summing log-odds would treat the
predictors as independent sources of evidence and drive agreement towards
certainty, but they share a base model, a prompt and a training signal, so their
errors are strongly correlated and their agreement is close to uninformative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Weight on the fine-tuned predictor's log-odds, with (1 - w) on the zero-shot.
# Shipping at 1.0 -- defer entirely to the fine-tuned model, which is the
# measured better predictor on the held-out split. Fitting w needs a set where
# both predictors answer, which is a measurement rather than a code change; when
# it is fitted the value replaces this constant and nothing else moves.
DEFAULT_W = 1.0

# Probabilities are clipped before the logit so a saturated input cannot produce
# an infinite log-odds. The recorded fine-tuned probabilities include values
# below 1e-7, so this is reached in practice, not defensively.
_EPS = 1e-6

# A source contributes to pooling only when it reports this status. Every other
# status -- a statement the fine-tuned model has no recorded probability for, a
# timeout, a dead agent, a short batch -- lands in the same branch, so a live
# fine-tuned endpoint removes a condition rather than changing the logic.
OK = "ok"


@dataclass
class SourceResult:
    """One predictor's answer for one statement."""

    status: str                       # ok | unavailable | timeout | error | not_returned
    probability: float | None = None
    prediction: int | None = None
    reason: str = ""
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status == OK and self.probability is not None

    def to_dict(self) -> dict:
        out: dict = {"status": self.status}
        if self.usable:
            out["probability"] = round(float(self.probability), 6)
            out["prediction"] = int(self.prediction)
        else:
            out["probability"] = None
            out["prediction"] = None
            if self.reason:
                out["reason"] = self.reason
            if self.detail:
                out["detail"] = self.detail[:400]
        return out


def logit(p: float) -> float:
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pool(zero_shot: float, fine_tuned: float, w: float = DEFAULT_W) -> float:
    """Weighted log-odds pool of two calibrated probabilities."""
    return sigmoid((1.0 - w) * logit(zero_shot) + w * logit(fine_tuned))


def reconcile(zero_shot: SourceResult, fine_tuned: SourceResult,
              *, w: float = DEFAULT_W, threshold: float = 0.5) -> dict:
    """One statement's verdict, and an account of how it was reached.

    A verdict backed by one predictor is still a verdict. It is distinguished by
    `reconciliation.applied` being false and by the other source's status, both
    of which are always present -- a reader who does not go looking for a
    missing field would otherwise assume both predictors answered.
    """
    usable = {"zero_shot": zero_shot, "fine_tuned": fine_tuned}
    available = {k: v for k, v in usable.items() if v.usable}

    if len(available) == 2:
        probability = pool(zero_shot.probability, fine_tuned.probability, w)
        reconciliation = {
            "method": "log_odds_pool", "w": w, "applied": True,
            "reason": "both_sources",
            "sources_used": ["zero_shot", "fine_tuned"],
            "detail": (f"Pooled the two calibrated probabilities in log-odds with "
                       f"weight {w} on the fine-tuned predictor."),
        }
    elif len(available) == 1:
        name, source = next(iter(available.items()))
        probability = float(source.probability)
        other = "fine_tuned" if name == "zero_shot" else "zero_shot"
        why = usable[other].reason or usable[other].status
        reconciliation = {
            "method": "log_odds_pool", "w": w, "applied": False,
            "reason": "single_source", "sources_used": [name],
            "detail": (f"Pooling did not run: the {other.replace('_', '-')} predictor "
                       f"did not return a probability for this statement ({why}). "
                       f"The verdict is the {name.replace('_', '-')} calibrated "
                       f"probability alone."),
        }
    else:
        return {
            "verdict": None, "probability": None, "status": "no_verdict",
            "agreement": {"zero_shot": None, "fine_tuned": None},
            "sources": {"zero_shot": zero_shot.to_dict(),
                        "fine_tuned": fine_tuned.to_dict()},
            "reconciliation": {
                "method": "log_odds_pool", "w": w, "applied": False,
                "reason": "no_source", "sources_used": [],
                "detail": "Neither predictor returned a probability for this statement.",
            },
        }

    verdict = bool(probability >= threshold)
    return {
        "verdict": verdict,
        "probability": round(float(probability), 6),
        "status": "ok",
        # Per-source booleans, so a predictor that did not answer reads as null
        # rather than as a prediction of False.
        "agreement": {
            "zero_shot": bool(zero_shot.prediction) if zero_shot.usable else None,
            "fine_tuned": bool(fine_tuned.prediction) if fine_tuned.usable else None,
        },
        "sources": {"zero_shot": zero_shot.to_dict(),
                    "fine_tuned": fine_tuned.to_dict()},
        "reconciliation": reconciliation,
    }

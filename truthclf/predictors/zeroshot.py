"""Zero-shot truthfulness predictor.

Two elicitation modes:

DEFAULT (use_logprobs=True): a single-token True/False answer whose token
logprobs give p(True) = softmax over the True/False tokens. This is the reported
baseline because it is token-based and therefore directly comparable to the
fine-tuned model (same elicitation on both sides).

SECONDARY (use_logprobs=False): the model returns a 0-100 score; score/100 is the
probability and score >= 50 -> True. On a parse failure the score is set to a
neutral 50 and a parse-failure counter is incremented. This mode produces
*continuous* probabilities, which is why the explainer uses it for occlusion
(removing a field then yields a graded probability shift); it is not directly
comparable to the token-based fine-tuned model.
"""

from __future__ import annotations

import math
import random
import re

from .. import prompts
from .base import Predictor, PredictionResult, compute_metrics

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def parse_score(text: str):
    """Extract the first 0-100 number from model output, clamped to [0, 100].
    Returns a float, or None if no number is present."""
    if not text:
        return None
    m = _NUMBER.search(text)
    if not m:
        return None
    return max(0.0, min(100.0, float(m.group())))


def prob_from_logprobs(top_logprobs: dict):
    """Map first-token logprobs to p(True) via softmax over the True/False
    tokens. Returns None if neither appears (caller applies a fallback)."""
    p_true = p_false = None
    for token, lp in top_logprobs.items():
        t = token.strip().lower()
        if t.startswith("true"):
            p = math.exp(lp)
            p_true = p if p_true is None else max(p_true, p)
        elif t.startswith("false"):
            p = math.exp(lp)
            p_false = p if p_false is None else max(p_false, p)
    if p_true is None and p_false is None:
        return None
    if p_true is None:
        return max(0.0, min(1.0, 1.0 - p_false))
    if p_false is None:
        return max(0.0, min(1.0, p_true))
    return p_true / (p_true + p_false)


class ZeroShotPredictor(Predictor):
    def __init__(self, model, variant="full", threshold=0.5, client=None, seed=0,
                 use_logprobs=True, neutral_score=50.0, calibrator=None):
        """`calibrator` is an optional DecisionArtifact (or a path to one).

        When supplied, predict() returns CALIBRATED probabilities and decides at
        the artifact's tuned threshold. When omitted — the default — predict()
        returns RAW probabilities decided at `threshold` (0.5), exactly as
        before: loading an artifact is opt-in, so nothing changes behaviour
        silently. The artifact is checked against `model` at construction and
        raises CalibratorModelMismatch on a mismatch."""
        self.model = model
        self.calibrator = self._load_calibrator(calibrator, model)
        self.variant = variant
        self.threshold = threshold
        self.client = client
        self.use_logprobs = use_logprobs
        self.neutral_score = neutral_score
        self._rng = random.Random(seed)

    @staticmethod
    def _load_calibrator(calibrator, model):
        if calibrator is None:
            return None
        from ..evaluation import DecisionArtifact
        art = (DecisionArtifact.load(calibrator) if isinstance(calibrator, str)
               else calibrator)
        art.check_model(model)          # hard error, never a warning
        return art

    def _messages(self, rows, mode):
        return [prompts.build_messages(r, self.variant, mode=mode) for r in rows]

    def rationale(self, rows, max_tokens=64) -> list:
        """One-sentence natural-language rationale per row (a plausible, not
        necessarily faithful, explanation). Uses the same client/model."""
        if self.client is None:
            raise RuntimeError("no client configured for rationale()")
        return self.client.complete(self._messages(rows, "rationale"), max_tokens=max_tokens)

    def _predict_score(self, rows):
        raw = self.client.score(self._messages(rows, "score"))
        scores, parse_failures = [], 0
        for text in raw:
            s = parse_score(text)
            if s is None:
                parse_failures += 1
                s = self.neutral_score
            scores.append(s)
        probs = [s / 100.0 for s in scores]
        preds = [1 if s >= 50.0 else 0 for s in scores]
        return scores, probs, preds, parse_failures

    def _predict_logprobs(self, rows):
        outputs = self.client.classify(self._messages(rows, "decision"))
        probs, parse_failures = [], 0
        for out in outputs:
            p = prob_from_logprobs(out.get("top_logprobs", {}))
            if p is None:
                parse_failures += 1
                p = 0.5
            probs.append(p)
        preds = [1 if p >= self.threshold else 0 for p in probs]
        return None, probs, preds, parse_failures

    def predict(self, points, labels=None) -> PredictionResult:
        if self.client is None:
            raise RuntimeError(
                "ZeroShotPredictor has no client configured. Pass an LLM client "
                "(e.g. truthclf.llm.TogetherClient) to make scored predictions."
            )
        rows = points          # the spec (and FinetunedPredictor) name this `points`
        if self.use_logprobs:
            scores, probs, preds, parse_failures = self._predict_logprobs(rows)
        else:
            scores, probs, preds, parse_failures = self._predict_score(rows)

        threshold_used = self.threshold
        if self.calibrator is not None:
            probs, preds = self.calibrator.decide(probs)
            threshold_used = self.calibrator.threshold

        result_metrics = compute_metrics(labels, preds, probs) if labels is not None else None
        return PredictionResult(
            scores=scores, probs=probs, preds=preds, threshold=threshold_used,
            parse_failures=parse_failures, n=len(rows), metrics=result_metrics,
        )

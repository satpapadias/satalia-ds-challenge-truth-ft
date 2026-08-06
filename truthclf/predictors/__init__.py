"""Predictors: a shared interface plus zero-shot and fine-tuned implementations."""

from __future__ import annotations

from .base import Predictor, PredictionResult
from .zeroshot import ZeroShotPredictor, parse_score
from .finetuned import FinetunedPredictor

__all__ = ["Predictor", "PredictionResult", "ZeroShotPredictor", "parse_score",
           "FinetunedPredictor"]

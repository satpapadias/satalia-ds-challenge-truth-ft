"""truthclf — binary truthfulness classification of short public statements.

Provides the data pipeline, evaluation metrics, calibration/threshold tuning,
zero-shot and fine-tuned predictors, and a faithfulness-aware explainer. See the
README for usage; import submodules directly (e.g. ``from truthclf import data``).
"""

from __future__ import annotations

# so any script that imports this package can read them via os.getenv(...).
# Searches from the current working directory upward; no-op if .env is absent.
#
# python-dotenv is a hard dependency, so this import is not guarded. It used to
# sit behind `except ImportError: pass`, which claimed the dependency was
# optional; in practice that only meant a broken install would surface later as
# a confusing "no API key" error instead of an honest ImportError here.
from dotenv import load_dotenv

load_dotenv()

from . import data, metrics

__all__ = ["data", "metrics"]

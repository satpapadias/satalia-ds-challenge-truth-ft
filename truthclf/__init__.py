"""truthclf — binary truthfulness classification of short public statements.

Provides the data pipeline, evaluation metrics, calibration/threshold tuning,
zero-shot and fine-tuned predictors, and a faithfulness-aware explainer. See the
README for usage; import submodules directly (e.g. ``from truthclf import data``).
"""

from __future__ import annotations

# Load secrets (e.g. TOGETHER_API_KEY) from a project-root .env at import time,
# so any script that imports this package can read them via os.getenv(...).
# Searches from the current working directory upward; no-op if .env is absent.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import data, metrics

__all__ = ["data", "metrics"]

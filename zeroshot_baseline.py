"""Calibrated zero-shot baseline on the held-out test split, using the
logprob/decision elicitation (the reported baseline, comparable to the fine-tuned
model). Serverless — no dedicated endpoint, ~cents. Platt calibration and the
decision threshold are fit on validation and reported on the untouched test split.

Expected: accuracy ~0.668, ECE ~0.061 equal-mass (NOT the score-mode 0.700).
"""

from __future__ import annotations

import numpy as np

from truthclf import data, experiments, calibration, threshold, metrics

BASE = "google/gemma-4-31B-it"
VARIANT = "full"
SCHEME = "primary"


def main():
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    _, val, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    vy = [r.y(SCHEME) for r in val]
    ty = [r.y(SCHEME) for r in test]

    # logprob/decision elicitation on both sides, via the Together Batch API
    # (single job per split — minutes, not a serial per-row loop)
    vp, _ = experiments.run_on_rows(BASE, VARIANT, val, SCHEME, use_logprobs=True, backend="batch")
    tp, _ = experiments.run_on_rows(BASE, VARIANT, test, SCHEME, use_logprobs=True, backend="batch")

    cal = calibration.fit_best(vp, vy)               # Platt/temperature fit on val
    tpc = list(calibration.apply(tp, cal))
    thr, _ = threshold.tune_threshold(list(calibration.apply(vp, cal)), vy,
                                      "balanced_accuracy")
    preds = (np.asarray(tpc) >= thr).astype(int)
    m = metrics.metric_bundle(ty, preds, tpc)

    print(f"Zero-shot logprob baseline — {BASE} [{VARIANT}], test n={len(ty)}")
    print(f"  calibrator={cal['method']}  threshold={thr:.3f}")
    print(f"  accuracy={m['accuracy']:.3f}  balanced_acc={m['balanced_accuracy']:.3f}  "
          f"macro_f1={m['macro_f1']:.3f}  brier={m['brier']:.3f}  ece={m['ece']:.3f}")


if __name__ == "__main__":
    main()

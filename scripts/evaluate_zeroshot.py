"""Threshold tuning + calibration + metadata ablation + selective prediction
for the lead config.

Runs in SCORE mode (use_logprobs=False) on purpose: threshold sweeps, reliability,
and the abstention curve need continuous probabilities, which the 0-100 score
gives and the token/logprob mode does not. This is therefore the score-mode
secondary study; the headline zero-shot baseline (logprob/decision, comparable to
the fine-tuned model) is produced by evaluate_finetuned.py.

Tuning and calibration are fit on a speaker-disjoint validation split; final
numbers are reported on the untouched test split.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import experiments, selective

LEAD_MODEL = "gemini-2.5-flash"
LEAD_VARIANT = "full"


def hr(title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


def main():
    res = experiments.phase_a_eval(LEAD_MODEL, LEAD_VARIANT)
    print(f"Lead: {LEAD_MODEL} [{LEAD_VARIANT}]  "
          f"val n={len(res.val_labels)}  test n={len(res.test_labels)}")

    hr("1. THRESHOLD TUNING (tuned on val, reported on TEST)")
    print(res.threshold_table.to_string(index=False))
    print("\n  -> threshold tuned on validation; test is untouched. Cost row makes a"
          " FALSE POSITIVE (a false statement labelled True) costlier than a false"
          " negative — the misinformation-relevant error in this 1=True encoding.")

    hr("2. CALIBRATION (fit on val, reported on TEST)")
    print(f"  calibrator chosen: {res.calibrator}")
    print(res.calibration_table.to_string(index=False))
    print("\n  -> source probability was the uncalibrated score/100.")

    hr("3. METADATA ABLATION (TEST split): statement-only vs full")
    print(experiments.ablation_table().to_string(index=False))

    hr("4. SELECTIVE PREDICTION (TEST, raw-score confidence): coverage vs accuracy")
    ty, tp = res.test_labels, res.test_probs
    for cov in (1.0, 0.9, 0.8, 0.7, 0.5):
        acc = selective.accuracy_at_coverage(tp, ty, cov)
        print(f"   coverage {cov:>4.0%}  ->  accuracy {acc:.3f}")
    print("\n  -> confidence = |score/100 - 0.5|; calibration compresses the probs,"
          " so raw score distance is the cleaner confidence signal for abstention.")

    hr("TUNED + CALIBRATED ZERO-SHOT, SCORE MODE (TEST)")
    o = res.official
    m = o["metrics"]
    print(f"  config: {LEAD_MODEL} [{LEAD_VARIANT}]  "
          f"calibrator={res.calibrator['method']}  threshold={o['threshold']:.3f}")
    print(f"  accuracy={m['accuracy']:.3f}  balanced_acc={m['balanced_accuracy']:.3f}  "
          f"macro_f1={m['macro_f1']:.3f}  brier={m['brier']:.3f}  ece={m['ece']:.3f}")
    print("\n  This is the SCORE-MODE (secondary) calibrated result. The headline"
          " zero-shot baseline compared against fine-tuning uses the logprob/decision"
          " elicitation — see evaluate_finetuned.py.")

    hr("calibration + threshold tuning complete")


if __name__ == "__main__":
    main()

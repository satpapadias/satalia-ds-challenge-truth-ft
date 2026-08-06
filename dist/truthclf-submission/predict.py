"""Zero-shot truthfulness prediction on a set of statements.

Demonstrates the predictor interface: predict(points, labels=None) processes a
SET of rows and returns per-point predictions/probabilities plus evaluation
metrics when labels are supplied. Uses the Together serverless model; responses
are cached on disk so reruns are free.

Usage:  python3 predict.py            # 20 held-out rows, gemma-4-31B-it [full]
"""

from __future__ import annotations

from truthclf import experiments

MODEL = "google/gemma-4-31B-it"
VARIANT = "full"
N = 20


def main():
    predictor = experiments.zeroshot_predictor(MODEL, variant=VARIANT, backend="sync")
    rows = experiments.sample_rows("test", n=N, seed=0)
    labels = [r.y("primary") for r in rows]

    result = predictor.predict(rows, labels=labels)

    print(f"Zero-shot predictions — {MODEL} [{VARIANT}], {result.n} rows "
          f"(parse failures: {result.parse_failures})\n")
    print(f"  {'pred':<6}{'P(True)':>8}{'label':>7}   statement")
    print("  " + "-" * 70)
    for row, pred, prob, y in list(zip(rows, result.preds, result.probs, labels))[:10]:
        print(f"  {'True' if pred else 'False':<6}{prob:>8.2f}{y:>7}   {row.statement[:48]}")

    m = result.metrics
    print(f"\n  metrics on this set: acc={m['accuracy']:.3f} "
          f"balanced_acc={m['balanced_accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
          f"brier={m['brier']:.3f}")


if __name__ == "__main__":
    main()

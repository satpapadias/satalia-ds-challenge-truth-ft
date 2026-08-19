"""Full-test zero-shot evaluation.

Runs both candidate bases x both prompt variants over the FULL cleaned dataset
using the batch backend (cheaper, no concurrency code), caches all responses,
and reports the first proper evaluation table: accuracy, balanced accuracy,
macro-F1, per-class precision/recall, parse/error-fail %, Brier and ECE — each
with bootstrap confidence intervals.

Uses the 0-100 score elicitation (secondary mode) for model selection across
candidate bases. Responses are cached so reruns are free.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import experiments, metrics

SCHEME = "primary"
MODELS = ["gemini-2.5-flash"]
VARIANTS = ["statement_only", "full"]
N_BOOT = 1000
DATA_PATH = "data.csv"

METRIC_ORDER = [
    "accuracy", "balanced_accuracy", "macro_f1",
    "precision_True", "recall_True", "precision_False", "recall_False",
    "roc_auc", "pr_auc", "brier", "ece",
]


def hr(title):
    print(f"\n{'='*78}\n  {title}\n{'='*78}", flush=True)


def main():
    runs = []
    for model in MODELS:
        for variant in VARIANTS:
            print(f"\n>>> running {model} [{variant}] via Batch API ...", flush=True)
            run = experiments.run_zeroshot(model, variant=variant, split="full",
                                           scheme=SCHEME, data_path=DATA_PATH,
                                           backend="batch")
            boot = metrics.bootstrap_bundle(run.labels, run.preds, run.probs, n_boot=N_BOOT)
            runs.append((run, boot))
            print(f"    done: n={run.n} parse_fail={run.parse_failures} "
                  f"error_fail={run.error_failures}", flush=True)

    hr(f"FULL-TEST RESULTS (n={runs[0][0].n}, primary labels, 95% bootstrap CIs)")
    for run, boot in runs:
        n = run.n
        pf = (run.parse_failures + run.error_failures) / n * 100
        print(f"\n  {run.model.split('/')[-1]}  [{run.variant}]   "
              f"parse/error-fail = {run.parse_failures}+{run.error_failures} "
              f"({pf:.1f}%)")
        for m in METRIC_ORDER:
            pt, lo, hi = boot[m]
            print(f"      {m:<18} {pt:6.3f}  [{lo:6.3f}, {hi:6.3f}]")

    hr("COMPACT SUMMARY (point estimates)")
    head = (f"  {'model':<16}{'variant':<15}{'acc':>7}{'bal_acc':>9}"
            f"{'macroF1':>9}{'brier':>8}{'ece':>7}{'fail%':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for run, boot in runs:
        pf = (run.parse_failures + run.error_failures) / run.n * 100
        print(f"  {run.model.split('/')[-1]:<16}{run.variant:<15}"
              f"{boot['accuracy'][0]:>7.3f}{boot['balanced_accuracy'][0]:>9.3f}"
              f"{boot['macro_f1'][0]:>9.3f}{boot['brier'][0]:>8.3f}"
              f"{boot['ece'][0]:>7.3f}{pf:>6.1f}%")

    hr("full-test table complete")


if __name__ == "__main__":
    main()

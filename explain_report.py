"""Explainer (Component 3) aggregate report on the zero-shot serverless predictor
(no endpoint). Runs explain() on a ~300-row test sample and reports the field-flip
table, driver distribution, and the rationale-vs-occlusion agreement rate.

Uses score-mode (use_logprobs=False): occlusion needs continuous probabilities so
that removing a field produces a graded shift rather than a 0/1 jump.
"""

from __future__ import annotations

import json
import random

from truthclf import data, explain, experiments

BASE = "google/gemma-4-31B-it"
SCHEME = "primary"
N_SAMPLE = 300
SEED = 0
RESULTS = "explain_results.json"


def hr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}", flush=True)


def main():
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    _, _, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    sample = random.Random(SEED).sample(test, min(N_SAMPLE, len(test)))
    labels = [r.y(SCHEME) for r in sample]

    hr(f"RUNNING explain() on {len(sample)} test rows (Batch API)")
    model = experiments.zeroshot_predictor(BASE, "full", backend="batch", use_logprobs=False)
    result = explain.explain(model, sample, labels=labels, with_rationale=True)
    agg = explain.aggregate(result)

    hr("AGGREGATE FIELD IMPORTANCE (how often removing a field flips the prediction)")
    print(agg["field_table"].to_string(index=False))
    print(f"\n  driver distribution: {agg['driver_distribution']}")
    print(f"  rationale-vs-occlusion agreement rate: "
          f"{agg['rationale_occlusion_agreement_rate']}")
    if "metrics" in result:
        m = result["metrics"]
        print(f"  base predictions on sample: acc={m['accuracy']:.3f} "
              f"bal_acc={m['balanced_accuracy']:.3f}")

    hr("EXAMPLE EXPLANATIONS (first 4)")
    for pp in result["per_point"][:4]:
        print(f"\n  [{pp['row_id']}] pred={'True' if pp['pred'] else 'False'} "
              f"(p={pp['base_prob']}) label={pp['label']}")
        print(f"    statement: {pp['statement'][:90]}")
        deltas = ", ".join(f"{f}:{pp['occlusion'][f]['delta']:+.2f}"
                           f"{'(FLIP)' if pp['occlusion'][f]['flip'] else ''}"
                           for f in explain.OCCLUSION_FIELDS)
        print(f"    occlusion Δp: {deltas}")
        print(f"    occlusion driver: {pp['driver']}")
        print(f"    rationale: {pp['rationale']}")
        print(f"    rationale cites: {pp['rationale_refs']}  -> "
              f"agree with occlusion: {pp['agree']}")

    json.dump({"aggregate": {k: (v.to_dict("records") if hasattr(v, "to_dict") else v)
                             for k, v in agg.items()},
               "metrics": result.get("metrics"),
               "examples": result["per_point"][:8]},
              open(RESULTS, "w"), indent=2, default=float)
    print(f"\n  results written to {RESULTS}")
    hr("explainer aggregate complete")


if __name__ == "__main__":
    main()

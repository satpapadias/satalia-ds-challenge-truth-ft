"""Full-test zero-shot cost + token estimate (NO API calls).

Estimates the cost of running zero-shot over the FULL cleaned dataset for both
candidate bases x both prompt variants. Zero-shot has no training, so all cleaned
rows are usable as the evaluation set; the later fine-tuned comparison will be
reported on the speaker-disjoint test split (a subset of these rows).

Prints the estimate and STOPS — approval required before any full run.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data, prompts, llm

DATA_PATH = "data.csv"
SCHEME = "primary"
VARIANTS = ["statement_only", "full"]
MODELS = ["gemini-2.5-flash"]
EST_OUTPUT = 4          # thinking disabled -> a 0-100 score is ~3-4 tokens


def hr(title):
    print(f"\n{'='*76}\n  {title}\n{'='*76}")


def main():
    rows = data.load(DATA_PATH)
    clean, _ = data.clean_dataset(rows, SCHEME)
    n = len(clean)
    print(f"Full cleaned set: {n} rows")
    print(f"Tokenizer: {llm.TOKENIZER} | est output tokens/call: {EST_OUTPUT}")

    in_toks = {v: sum(llm.count_tokens(prompts.render(r, v, "score")) for r in clean)
               for v in VARIANTS}

    hr("FULL-TEST TOKENS (per variant; model-independent)")
    for v in VARIANTS:
        print(f"  {v:<16} input={in_toks[v]:>11,}  output={n*EST_OUTPUT:>9,}")

    hr("FULL-TEST COST ESTIMATE (USD)")
    print(f"  {'model':<22}{'variant':<16}{'input$':>10}{'output$':>10}{'total$':>10}")
    print("  " + "-" * 66)
    grand = 0.0
    for model in MODELS:
        price = llm.PRICING[model]
        for v in VARIANTS:
            in_cost = in_toks[v] * price["input"] / 1_000_000
            out_cost = n * EST_OUTPUT * price["output"] / 1_000_000
            total = in_cost + out_cost
            grand += total
            print(f"  {model.split('/')[-1]:<22}{v:<16}"
                  f"{in_cost:>9.3f}${out_cost:>9.3f}${total:>9.3f}$")
    print(f"\n  GRAND TOTAL (both models x both variants): ${grand:.3f}")
    per_model = {m: sum((in_toks[v] * llm.PRICING[m]["input"]
                         + n * EST_OUTPUT * llm.PRICING[m]["output"]) / 1_000_000
                        for v in VARIANTS) for m in MODELS}
    for m, c in per_model.items():
        print(f"    {m.split('/')[-1]}: ${c:.3f} (both variants)")

    hr("STOP — estimate only; awaiting approval before any full-test run")
    print("  Latency note: ~0.4s/call at temp 0; ~4 runs of "
          f"{n} rows. Cached, so a rerun is free.")


if __name__ == "__main__":
    main()

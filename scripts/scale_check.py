"""Zero-shot scale check on a 200-row dev subset (Together AI).

Runs the three real fine-tune-base candidates (each is both serverless and
LoRA-fine-tunable under a SINGLE id — no Reference/Turbo confound) and reports a
combined table that also includes the previously cached context models
(gpt-oss-20b, Llama-3.3-70B-Turbo). Prints a cost estimate first; responses are
cached so reruns are free.

Reasoning models are configured in llm.MODEL_CONFIGS: gpt-oss uses
reasoning_effort='low' + a larger budget; Qwen/Gemma disable thinking.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data, prompts, llm, experiments

DATA_PATH = "data.csv"
SCHEME = "primary"
DEV_N = 200
SEED = 0
VARIANTS = ["statement_only", "full"]

# Real fine-tune-base candidates (this round).
CANDIDATES = ["Qwen/Qwen3.5-9B", "google/gemma-4-31B-it", "openai/gpt-oss-120b"]
# Context only (already cached from the previous round).
CONTEXT = ["openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]

# Rough output-token budget per call, for the pre-run cost estimate.
EST_OUTPUT = {
    "Qwen/Qwen3.5-9B": 3, "google/gemma-4-31B-it": 3,
    "openai/gpt-oss-120b": 400, "openai/gpt-oss-20b": 100,
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": 2,
}


def hr(title):
    print(f"\n{'='*82}\n  {title}\n{'='*82}")


def main():
    rows = data.load(DATA_PATH)
    clean, _ = data.clean_dataset(rows, SCHEME)
    dev = data.dev_subset(clean, n=DEV_N, seed=SEED)
    n = len(dev)
    print(f"{len(clean)} cleaned rows; dev subset = {n} "
          f"(True {sum(r.y(SCHEME) for r in dev)})")
    print(f"Tokenizer: {llm.TOKENIZER}")

    in_toks = {v: sum(llm.count_tokens(prompts.render(r, v, 'score')) for r in dev)
               for v in VARIANTS}

    hr("COST ESTIMATE (candidates, dev subset, before any new call)")
    grand = 0.0
    print(f"  {'model':<26}{'variant':<16}{'est USD':>10}")
    print("  " + "-" * 50)
    for model in CANDIDATES:
        for v in VARIANTS:
            cost = llm.estimate_cost(in_toks[v], n * EST_OUTPUT[model], model)
            grand += cost
            print(f"  {model.split('/')[-1]:<26}{v:<16}{cost:>9.4f}$")
    print(f"\n  GRAND TOTAL (candidates x variants): ${grand:.4f}")

    hr("RUNNING (candidates make calls; context models served from cache)")
    runs = []
    for model in CANDIDATES + CONTEXT:
        for v in VARIANTS:
            run = experiments.run_zeroshot(model, variant=v, n_rows=DEV_N,
                                           split="dev", scheme=SCHEME, seed=SEED,
                                           data_path=DATA_PATH)
            runs.append(run)
            print(f"  done: {model.split('/')[-1]:<28}[{v:<14}] "
                  f"parse_fail={run.parse_failures} latency={run.avg_latency_s:.1f}s")

    hr("COMBINED RESULTS (dev subset, 200 rows) — candidates first, context last")
    print(experiments.metrics_table(runs).to_string(index=False))

    hr("STOP — report only; no §3.1 lock, no fine-tune, no full-test run")


if __name__ == "__main__":
    main()

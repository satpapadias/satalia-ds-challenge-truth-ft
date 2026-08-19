"""One-shot migration of the schema-1 JSON response cache into schema-2 diskcache.

Schema 1 (`.llm_cache.json`) hashed only (model, messages, params). The serving
backend was deliberately excluded so that sync and Batch-API runs would share
entries — which meant the two paths could answer the same prompt differently and
whichever ran last silently overwrote the other. Schema 2 puts `backend` and
`call` in the key and stores a provenance envelope.

Migration policy (approved before running):

  MIGRATE    entries attributable to exactly ONE backend, whose value shape is
             still valid under schema 2 (score / complete: raw model text).
  QUARANTINE entries reachable from BOTH a sync and a batch entrypoint — their
             stored value cannot be attributed to a backend, so it is left
             behind and refetched lazily under a known one.
  QUARANTINE every `classify` entry — schema 1 stored the FLATTENED
             {token: logprob} map, schema 2 stores the raw API structure and
             parses on read. The old shape cannot be reconstructed.
  DROP       865 orphans (unreachable by any prompt the current code builds,
             left over from a superseded prompt template) and the 60
             empty-string failure sentinels.

Nothing is deleted: `.llm_cache.json` is left byte-for-byte untouched as the
v1 archive. Migration makes no API calls and costs nothing; quarantined keys
refetch on the next evaluation run.

    python3 scripts/migrate_cache.py --dry-run     # report only (default)
    python3 scripts/migrate_cache.py --apply       # write .llm_cache/
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data, explain, llm, prompts  # noqa: E402

V1_PATH = ".llm_cache.json"
SCHEME, DATA_PATH = "primary", "data.csv"
GEMMA = "gemini-2.5-flash"
QWEN = "Qwen/Qwen3.5-9B"


def _cfg(model):
    c = llm.MODEL_CONFIGS.get(model, llm.DEFAULT_CONFIG)
    return c.get("max_output_tokens", 16), {
        "temperature": 0.0,
        "reasoning_effort": c.get("reasoning_effort"),
        "extra": c.get("extra_create_kwargs", {}),
    }


_v1_key = llm.legacy_v1_key                 # frozen schema-1 format, shared


def _requests():
    """Every (backend, call, model, variant, row) the recorded entrypoints issued.

    Yields (backend, call, model, messages, extra_params). Mirrors the scripts as
    they exist in git: fulltest_run / zeroshot_baseline / eval_finetune /
    phase_a_report / explain_report are batch; scale_check / explain_analysis /
    predict are sync.
    """
    rows = data.load(DATA_PATH)
    clean, _ = data.clean_dataset(rows, SCHEME)
    _, val, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    dev200 = data.dev_subset(clean, 200, 0)
    sample = random.Random(0).sample(test, 300)

    occl = []
    for p in sample:
        occl.append(p)
        for f in explain.OCCLUSION_FIELDS:
            occl.append(dataclasses.replace(p, **{f: ""}))
        occl.append(dataclasses.replace(p, **{f: "" for f in explain.ALL_META}))

    def score(model, rows_, variant, backend):
        mt, params = _cfg(model)
        for r in rows_:
            yield backend, "score", model, prompts.build_messages(r, variant, mode="score"), \
                dict(max_tokens=mt, **params)

    def classify(model, rows_, variant, backend):
        _, params = _cfg(model)
        for r in rows_:
            yield backend, "classify", model, \
                prompts.build_messages(r, variant, mode="decision"), \
                dict(max_tokens=1, logprobs=10, **params)

    def complete(model, rows_, variant, backend):
        _, params = _cfg(model)
        for r in rows_:
            yield backend, "complete", model, \
                prompts.build_messages(r, variant, mode="rationale"), \
                dict(max_tokens=64, kind="complete", **params)

    # --- batch entrypoints
    for m in (QWEN, GEMMA):                                    # benchmark_zeroshot.py
        for v in ("statement_only", "full"):
            yield from score(m, clean, v, "batch")
    yield from classify(GEMMA, val + test, "full", "batch")    # zeroshot_baseline, eval_finetune
    yield from score(GEMMA, val + test, "full", "batch")       # eval_finetune, phase_a_report
    yield from score(GEMMA, occl, "full", "batch")             # run_explainer.py
    yield from complete(GEMMA, sample, "full", "batch")
    # --- sync entrypoints
    for m in llm.PRICING:                                      # scale_check.py
        for v in ("statement_only", "full"):
            yield from score(m, dev200, v, "sync")
    yield from score(GEMMA, occl, "full", "sync")              # explain_analysis.py
    yield from complete(GEMMA, sample, "full", "sync")
    yield from score(GEMMA, data.dev_subset(test, 20, 0), "full", "sync")   # predict.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the schema-2 cache")
    args = ap.parse_args()

    with open(V1_PATH, encoding="utf-8") as f:
        v1 = json.load(f)
    print(f"schema-1 archive: {V1_PATH}  ({len(v1)} entries)\n")

    # Pass 1: attribute every v1 key to the backends/calls that can produce it.
    attrib: dict[str, set] = collections.defaultdict(set)
    plan: dict[str, tuple] = {}
    for backend, call, model, messages, params in _requests():
        k1 = _v1_key(model, messages, **params)
        if k1 in v1:
            attrib[k1].add((backend, call))
            plan[k1] = (backend, call, model, messages, params)

    stats = collections.Counter()
    migrate = []
    for k1, tags in attrib.items():
        backends = {b for b, _ in tags}
        calls = {c for _, c in tags}
        if v1[k1] == "":
            stats["drop: empty failure sentinel"] += 1
        elif len(backends) > 1:
            stats["quarantine: dual-backend, value unattributable"] += 1
        elif "classify" in calls:
            stats["quarantine: classify, pre-parsed value shape"] += 1
        else:
            stats["migrate"] += 1
            migrate.append(k1)
    unreachable = len(v1) - len(attrib)
    stats[f"drop: unreachable (superseded prompt)"] = unreachable

    print(f"{'disposition':<52}{'entries':>9}")
    print("-" * 61)
    for k, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<50}{n:>9}")
    print("-" * 61)
    print(f"  {'TOTAL':<50}{sum(stats.values()):>9}\n")

    if not args.apply:
        print("dry run — nothing written. Re-run with --apply to migrate.")
        return

    cache = llm.ResponseCache()
    for k1 in migrate:
        backend, call, model, messages, params = plan[k1]
        k2 = llm.ResponseCache.key(model, messages, backend=backend, call=call, **params)
        cache.set(k2, v1[k1], backend=backend, call=call, model=model)
    print(f"wrote {len(cache)} entries to {cache.path}")
    print(f"{V1_PATH} left untouched as the v1 archive.")


if __name__ == "__main__":
    main()

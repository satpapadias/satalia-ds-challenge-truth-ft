"""Refetch the cache entries quarantined by the schema-2 migration.

The migration (scripts/migrate_cache.py) refused to migrate two classes:
  * dual-backend entries, whose stored value could not be attributed to the sync
    or the Batch API path and so may have been overwritten by the wrong one;
  * every classify entry, stored in schema 1 as a pre-flattened
    {token: logprob} map that schema 2's raw-then-parse rule cannot reconstruct.

This fetches them again under a KNOWN backend (Batch API — the path the published
runs used), so every probability behind the corrected results has clean
provenance. Estimated cost ~$0.33, serverless, no dedicated endpoint.

    python3 scripts/refetch_quarantined.py
"""

from __future__ import annotations

import dataclasses
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data, explain, llm, prompts  # noqa: E402

G = "google/gemma-4-31B-it"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
SCHEME, VARIANT = "primary", "full"


def hr(t):
    print(f"\n{'=' * 78}\n  {t}\n{'=' * 78}", flush=True)


def main():
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    _, val, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    sample = random.Random(0).sample(test, 300)

    occl = []
    for p in sample:
        occl.append(p)
        for f in explain.OCCLUSION_FIELDS:
            occl.append(dataclasses.replace(p, **{f: ""}))
        occl.append(dataclasses.replace(p, **{f: "" for f in explain.ALL_META}))

    started = time.time()

    hr("(a) score-mode, val + test  [batch]")
    sc = llm.TogetherBatchClient(G, max_output_tokens=16,
                                 extra_create_kwargs=NO_THINK, poll_interval=20)
    sc.score([prompts.build_messages(r, VARIANT, mode="score") for r in val + test])
    print(f"  done: submitted={sc.n_submitted} errors={sc.error_count} "
          f"shape_errors={len(sc.shape_errors)}", flush=True)

    hr("(b) decision/logprob, val + test  [batch]")
    cc = llm.TogetherBatchClient(G, max_output_tokens=1,
                                 extra_create_kwargs=NO_THINK, poll_interval=20)
    cc.classify([prompts.build_messages(r, VARIANT, mode="decision") for r in val + test])
    print(f"  done: submitted={cc.n_submitted} errors={cc.error_count} "
          f"shape_errors={len(cc.shape_errors)}", flush=True)

    hr("(c) explainer: score over base + ablations  [batch]")
    ec = llm.TogetherBatchClient(G, max_output_tokens=16,
                                 extra_create_kwargs=NO_THINK, poll_interval=20)
    ec.score([prompts.build_messages(r, VARIANT, mode="score") for r in occl])
    print(f"  done: submitted={ec.n_submitted} errors={ec.error_count} "
          f"shape_errors={len(ec.shape_errors)}", flush=True)

    hr("(d) explainer: rationales  [batch]")
    rc = llm.TogetherBatchClient(G, max_output_tokens=64,
                                 extra_create_kwargs=NO_THINK, poll_interval=20)
    rc.complete([prompts.build_messages(r, VARIANT, mode="rationale") for r in sample],
                max_tokens=64)
    print(f"  done: submitted={rc.n_submitted} errors={rc.error_count} "
          f"shape_errors={len(rc.shape_errors)}", flush=True)

    total = sum(c.n_submitted for c in (sc, cc, ec, rc))
    errs = sum(c.error_count for c in (sc, cc, ec, rc))
    hr("REFETCH COMPLETE")
    print(f"  rows submitted : {total}")
    print(f"  failures       : {errs} (not cached — a rerun retries them)")
    print(f"  wall clock     : {(time.time() - started) / 60:.1f} min")
    print(f"  cache entries  : {len(llm.ResponseCache())}")
    for c, tag in ((sc, "score"), (cc, "classify"), (ec, "explainer-score"),
                   (rc, "rationale")):
        if c.error_samples:
            print(f"  !! {tag} error samples: {c.error_samples[:3]}")


if __name__ == "__main__":
    main()

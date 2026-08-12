"""Build the identity companion for the stored fine-tuned probabilities.

ft_eval_cache.json maps a row_id to a probability and nothing else. A row_id is
a position in data.csv, not a statement, so on its own it cannot tell whether
the probability being served was computed for the statement being asked about.
A caller that supplies its own statement under a row_id that happens to exist
receives another statement's probability, correctly calibrated and labelled as a
successful prediction.

This writes ft_eval_identity.json: row_id -> the normalised statement key of the
statement the probability was actually computed for. The serving path compares
both, so a mismatch is reported as unavailable rather than answered.

Deterministic: identical inputs produce identical bytes. Rebuild with

    python scripts/build_ft_identity.py

and commit the result alongside the probabilities it describes.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthclf import data  # noqa: E402

CACHE = "ft_eval_cache.json"
OUT = "ft_eval_identity.json"
SCHEME = "primary"


def main() -> int:
    with open(CACHE, encoding="utf-8") as f:
        cache = json.load(f)

    # The same cleaning stage the probabilities were produced under, so row_ids
    # line up with the split the evaluation ran on.
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    by_id = {r.row_id: r for r in clean}

    identity, orphans = {}, []
    for key in cache:
        row = by_id.get(int(key))
        if row is None:
            orphans.append(key)
            continue
        identity[key] = row.norm_key

    if orphans:
        # A stored probability whose row is no longer in the cleaned dataset
        # cannot be verified, so it must not be servable.
        print(f"WARNING: {len(orphans)} stored row_ids are absent from the "
              f"cleaned dataset and will be unservable: {orphans[:5]}")

    payload = {
        "_schema": 1,
        "_description": ("row_id -> normalised statement key for each stored "
                         "fine-tuned probability. Serving verifies both, so a "
                         "row_id reused for a different statement is refused."),
        "_source": CACHE,
        "_scheme": SCHEME,
        "n": len(identity),
        "identity": dict(sorted(identity.items(), key=lambda kv: int(kv[0]))),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0, sort_keys=False)
        f.write("\n")

    print(f"wrote {OUT}: {len(identity)} of {len(cache)} stored probabilities "
          f"bound to a statement")
    return 1 if orphans else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explainer follow-ups (cached, NO new API calls):
  1. spot-check the faithfulness field-mention rule (false-disagreement audit)
  2. cross dominant driver with correctness (does the speaker shortcut help/hurt?)
Re-runs explain() on the same 300-row sample via a SYNC client; every call must be
a cache hit (asserted via the client's api-call counter)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collections
import random

from truthclf import data, explain, experiments

BASE = "gemini-2.5-flash"
SCHEME = "primary"


def hr(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


def main():
    clean, _ = data.clean_dataset(data.load("data.csv"), SCHEME)
    _, _, test = data.speaker_disjoint_3way(clean, 0.2, 0.2, 0, SCHEME)
    sample = random.Random(0).sample(test, 300)
    by_id = {r.row_id: r for r in sample}
    labels = [r.y(SCHEME) for r in sample]

    model = experiments.zeroshot_predictor(BASE, "full", backend="sync")
    result = explain.explain(model, sample, labels=labels, with_rationale=True)
    calls = getattr(model.client, "n_api_calls", None)
    print(f"client API calls during re-analysis: {calls} (0 = fully cached, no spend)")

    per = result["per_point"]

    # ---- TASK 2: accuracy by dominant driver -------------------------------
    hr("DRIVER vs CORRECTNESS (accuracy by dominant occlusion driver)")
    by = collections.defaultdict(lambda: [0, 0])
    for pp in per:
        ok = int(pp["pred"] == pp["label"])
        by[pp["driver"]][0] += ok
        by[pp["driver"]][1] += 1
    print(f"  {'driver':<22}{'n':>5}{'accuracy':>10}")
    print("  " + "-" * 37)
    for d, (c, n) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {d:<22}{n:>5}{c/n:>10.3f}")
    # collapsed view
    def grp(d):
        return d if d in ("speaker_name", "statement") else "other_metadata"
    coll = collections.defaultdict(lambda: [0, 0])
    for pp in per:
        g = grp(pp["driver"])
        coll[g][0] += int(pp["pred"] == pp["label"]); coll[g][1] += 1
    print("\n  collapsed:")
    for g in ("statement", "speaker_name", "other_metadata"):
        c, n = coll[g]
        print(f"    {g:<16} n={n:>4}  acc={c/n:.3f}")

    # ---- TASK 1: false-disagreement spot-check -----------------------------
    hr("FAITHFULNESS SPOT-CHECK (5 disagreement cases — genuine vs false?)")
    diss = [pp for pp in per if pp["agree"] is False]
    print(f"  total disagreements: {len(diss)}/{len(per)}")
    for pp in diss[:5]:
        row = by_id[pp["row_id"]]
        rat = (pp["rationale"] or "").lower()
        spk_tokens = [w for w in (row.speaker_name or "").lower().replace("-", " ").split() if len(w) > 2]
        spk_in = any(w in rat for w in spk_tokens)
        print(f"\n  [{pp['row_id']}] driver={pp['driver']}  speaker={row.speaker_name!r}")
        print(f"    statement: {pp['statement'][:80]}")
        print(f"    rationale: {pp['rationale']}")
        print(f"    rationale_refs={pp['rationale_refs']}  | speaker-name token in rationale? {spk_in}")


if __name__ == "__main__":
    main()

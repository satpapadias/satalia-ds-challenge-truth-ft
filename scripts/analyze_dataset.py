"""
Exploratory data analysis for the truthfulness dataset.

Read-only analysis. No predictors, no LLM calls.
Reports:
  - 6-label distribution
  - post-binarization balance: PRIMARY (middle split) AND sensitivity (half-true -> False)
  - unique speaker count + top-20 speaker frequencies
  - % missing per column
  - affiliation value counts + truth-rate-by-affiliation (leakage signal)
  - count of likely-unjudgeable statements (justifies the abstention path)
  - encoding-damage scan (justifies the light cleaning step)
"""

from __future__ import annotations

import collections
import csv
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf.data import normalized_statement_key  # noqa: E402

DATA_PATH = "data.csv"

LABEL_ORDER = ["true", "mostly-true", "half-true", "barely-true", "false", "extremely-false"]

# Binary label mapping schemes (see README).
#   PRIMARY     = middle split: {true, mostly-true, half-true} -> True
#   SENSITIVITY = half-true -> False: {true, mostly-true} -> True
BINARY_MAPS = {
    "PRIMARY  -- middle split  ({true, mostly-true, half-true} -> True)": {
        "true": True, "mostly-true": True, "half-true": True,
        "barely-true": False, "false": False, "extremely-false": False,
    },
    "SENSITIVITY -- half-true->False ({true, mostly-true} -> True)": {
        "true": True, "mostly-true": True,
        "half-true": False, "barely-true": False, "false": False, "extremely-false": False,
    },
}
PRIMARY_MAP = BINARY_MAPS["PRIMARY  -- middle split  ({true, mostly-true, half-true} -> True)"]


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_of(row: dict) -> str:
    # Robust to "Label" vs "label" header casing.
    for k in ("Label", "label"):
        if k in row:
            return row[k].strip()
    raise KeyError("no label column found")


def section(title: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


def pct(n: int, total: int) -> str:
    return f"{n:>5}  ({n/total*100:5.1f}%)"


def percentile(data: list, p: float) -> float:
    """Linear-interpolation percentile (p in 0..100). numpy's default method
    ('linear') is exactly this; kept as a named wrapper for readability."""
    return float(np.percentile(data, p)) if len(data) else 0.0


# Statement grouping uses the SAME key as the package, imported rather than
# re-implemented: this script's numbers must describe the data the splitter
# actually sees. It previously carried its own copy, free to drift.
norm_statement = normalized_statement_key


def main() -> None:
    rows = load(DATA_PATH)
    total = len(rows)
    cols = list(rows[0].keys())

    section("DATASET OVERVIEW")
    print(f"Total statements : {total}")
    print(f"Columns          : {', '.join(cols)}")
    print("Shape            : 8 columns: label + statement + five speaker-metadata "
          "fields\n                   (subjects, speaker_name, speaker_job, "
          "speaker_state, speaker_affiliation) + statement_context.")

    # ------------------------------------------------------------------
    section("MISSING / EMPTY VALUES  (feed [unknown] for empty fields, not 'None')")
    for col in cols:
        empty = sum(1 for r in rows if not r[col].strip())
        bar = "!" if empty > 0 else " "
        print(f"  {bar} {col:<25} {empty:>5}  ({empty/total*100:4.1f}%)")
    print()
    print("  NOTE: speaker_job, speaker_state, statement_context missing in ~a third")
    print("  of rows. In prompts represent these with an explicit [unknown] token.")

    # ------------------------------------------------------------------
    section("6-WAY LABEL DISTRIBUTION")
    labels = [label_of(r) for r in rows]
    lc = collections.Counter(labels)
    for lbl in LABEL_ORDER:
        print(f"  {lbl:<20} {pct(lc[lbl], total)}")
    unexpected = set(lc) - set(LABEL_ORDER)
    if unexpected:
        print(f"  !! unexpected labels: {unexpected}")
    print()
    print("  Roughly uniform across 6 classes; 'extremely-false' is the small class.")

    # ------------------------------------------------------------------
    section("BINARY MAPPING -- CLASS BALANCE (primary + sensitivity)")
    for name, mapping in BINARY_MAPS.items():
        binary = [mapping[l] for l in labels]
        n_true = sum(binary)
        n_false = total - n_true
        ratio = n_true / n_false
        print(f"\n  {name}")
        print(f"    True  : {pct(n_true, total)}")
        print(f"    False : {pct(n_false, total)}")
        bal = "balanced" if 0.7 < ratio < 1.4 else "IMBALANCED"
        print(f"    Ratio T/F: {ratio:.2f}  ({bal})")
    print()
    print("  The PRIMARY middle split is near-balanced -> balanced-accuracy / macro-F1")
    print("  are well-behaved. The sensitivity split is ~35/65; report the metric shift.")

    # ------------------------------------------------------------------
    section("STATEMENT LENGTH (words + chars)")
    wlen = [len(r["statement"].split()) for r in rows]
    clen = [len(r["statement"]) for r in rows]
    pctiles = [50, 75, 90, 95, 99]
    print(f"  {'metric':<8} {'min':>5} {'mean':>7} {'max':>5}   "
          + "  ".join(f"p{p}" for p in pctiles))
    print(f"  {'words':<8} {min(wlen):>5} {statistics.mean(wlen):>7.1f} {max(wlen):>5}   "
          + "  ".join(f"{percentile(wlen, p):>3.0f}" for p in pctiles))
    print(f"  {'chars':<8} {min(clen):>5} {statistics.mean(clen):>7.1f} {max(clen):>5}   "
          + "  ".join(f"{percentile(clen, p):>3.0f}" for p in pctiles))
    print("\n  Word-length buckets:")
    for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 200)]:
        n = sum(1 for l in wlen if lo <= l <= hi)
        print(f"    {lo:>3}-{hi:<3} words : {pct(n, total)}")
    print("\n  Statements are short (mean ~18 words).")

    # ------------------------------------------------------------------
    section("REPEATED STATEMENTS (exact, and exact-after-normalisation)")
    # Exact duplicates (verbatim string match).
    stmt_rows = collections.defaultdict(list)
    for i, r in enumerate(rows):
        stmt_rows[r["statement"]].append(i)
    exact_dup_groups = {s: idxs for s, idxs in stmt_rows.items() if len(idxs) > 1}
    n_exact_dup_rows = sum(len(idxs) for idxs in exact_dup_groups.values())
    print(f"  Exact-duplicate statements : {len(exact_dup_groups)} groups "
          f"covering {n_exact_dup_rows} rows ({n_exact_dup_rows/total*100:.1f}%)")

    # Near-duplicates (normalized key) beyond exact matches.
    norm_rows = collections.defaultdict(list)
    for i, r in enumerate(rows):
        norm_rows[norm_statement(r["statement"])].append(i)
    repeat_groups = {s: idxs for s, idxs in norm_rows.items() if len(idxs) > 1}
    n_repeat_rows = sum(len(idxs) for idxs in repeat_groups.values())
    print(f"  Repeats after normalisation: {len(repeat_groups)} groups "
          f"covering {n_repeat_rows} rows ({n_repeat_rows/total*100:.1f}%)")
    print(f"    (extra rows caught by normalization only: "
          f"{n_repeat_rows - n_exact_dup_rows})")

    # Conflicting labels within a duplicate group -> label noise.
    conflict_groups = []
    for s, idxs in repeat_groups.items():
        lbls = {label_of(rows[i]) for i in idxs}
        if len(lbls) > 1:
            conflict_groups.append((s, idxs, lbls))
    print(f"\n  Duplicate groups with CONFLICTING labels : {len(conflict_groups)}")
    for s, idxs, lbls in conflict_groups[:6]:
        binset = {PRIMARY_MAP[label_of(rows[i])] for i in idxs}
        flips_binary = "  <-- flips binary label!" if len(binset) > 1 else ""
        short = (s[:54] + "...") if len(s) > 54 else s
        print(f"    {sorted(lbls)} {short!r}{flips_binary}")

    # Cross-split risk: a duplicated statement spoken by >1 distinct speaker
    # would land on BOTH sides of a speaker-disjoint split -> leakage.
    crosssplit = 0
    for s, idxs in repeat_groups.items():
        spk = {rows[i]["speaker_name"].strip() for i in idxs}
        if len(spk) > 1:
            crosssplit += 1
    print(f"\n  Repeat groups spanning >1 speaker (cross-split leak risk): {crosssplit}")
    print("  -> dedup BEFORE splitting, and assign each repeat group to ONE split side")
    print("     so identical text cannot leak across the speaker-disjoint train/test gap.")

    # ------------------------------------------------------------------
    section("VERY SHORT STATEMENTS  (minor tail; NOT the abstention rationale)")
    very_short = [r for r in rows if len(r["statement"].split()) < 5]
    short_nodigit = [r for r in very_short if not any(c.isdigit() for c in r["statement"])]
    print(f"  Statements < 5 words            : {pct(len(very_short), total)}")
    print(f"  ... and containing no digit     : {pct(len(short_nodigit), total)}")
    print("\n  Examples (< 5 words, no digit):")
    for r in short_nodigit[:8]:
        print(f"    [{label_of(r):<15}] {r['statement']!r}")
    print()
    print("  Only ~0.6% are literally too short to verify -> a MINOR tail, not a large")
    print("  'unjudgeable' population. Do NOT justify abstention with this. Abstention")
    print("  is framed around MODEL UNCERTAINTY: decline near p=0.5, and show accuracy")
    print("  rising as coverage drops (coverage-vs-accuracy curve).")

    # ------------------------------------------------------------------
    section("ENCODING / APOSTROPHE DAMAGE  (justifies light cleaning, S2)")
    damaged_tokens = ["didnt", "dont", "doesnt", "wont", "cant", "isnt", "wasnt",
                      "couldnt", "wouldnt", "shouldnt", "w:", "Childrens", "Obamas"]
    lowered = [r["statement"] for r in rows]
    n_damaged = 0
    hit_counter = collections.Counter()
    for s in lowered:
        toks = set(w.strip(".,!?;:\"'").lower() for w in s.split())
        damaged_here = [t for t in damaged_tokens if t.lower() in toks]
        if damaged_here:
            n_damaged += 1
            for t in damaged_here:
                hit_counter[t] += 1
    # Mangled dash heuristic: a stray ' - ' or non-ascii bytes.
    n_nonascii = sum(1 for s in lowered if any(ord(c) > 127 for c in s))
    print(f"  Rows with apostrophe-stripped contractions : {pct(n_damaged, total)}")
    print(f"  Rows containing non-ASCII characters       : {pct(n_nonascii, total)}")
    print("\n  Top damaged tokens:")
    for tok, cnt in hit_counter.most_common(8):
        print(f"    {tok:<12} {cnt:>5}")
    print()
    print("  Clean lightly (restore obvious contractions); do NOT over-normalize.")

    # ------------------------------------------------------------------
    section("SPEAKERS  (unique count + top-20; speaker-disjoint split rationale)")
    speakers = [r["speaker_name"].strip() or "(empty)" for r in rows]
    sc = collections.Counter(speakers)
    print(f"  Unique speakers : {len(sc)}")
    print(f"  Top-20 account for {sum(c for _, c in sc.most_common(20))/total*100:.1f}% of rows")
    print("\n  Top 20 speakers:")
    for spk, cnt in sc.most_common(20):
        print(f"    {spk:<35} {pct(cnt, total)}")
    print()
    print("  Heavy speaker concentration -> a stratified-random split LEAKS speaker")
    print("  identity. Use a speaker-disjoint split too; the gap = memorization estimate.")

    # ------------------------------------------------------------------
    section("NON-PERSON 'SPEAKERS'  (source-type shortcut risk, S3.7)")
    nonperson = ["facebook-posts", "chain-email", "blog-posting", "viral-image",
                 "tweets", "email", "instagram-posts"]
    print(f"  {'source':<20} {'count':>7}  {'False rate':>10}")
    print(f"  {'-'*20} {'-'*7}  {'-'*10}")
    for src in nonperson:
        subset = [r for r in rows if r["speaker_name"].strip() == src]
        if not subset:
            continue
        n_false = sum(1 for r in subset if not PRIMARY_MAP[label_of(r)])
        print(f"  {src:<20} {len(subset):>7}  {n_false/len(subset)*100:>9.1f}%")
    print()
    print("  These skew heavily False -> a model can 'predict' from source type alone.")
    print("  The explainer must expose whether predictions lean on this shortcut.")

    # ------------------------------------------------------------------
    section("AFFILIATION VALUE COUNTS")
    affiliations = [r["speaker_affiliation"].strip() or "(empty)" for r in rows]
    ac = collections.Counter(affiliations)
    for aff, cnt in ac.most_common(12):
        print(f"  {aff:<30} {pct(cnt, total)}")

    # ------------------------------------------------------------------
    section("TRUTH RATE BY AFFILIATION  (label-leakage signal, PRIMARY map)")
    for aff in ["republican", "democrat", "none", "organization"]:
        subset = [r for r in rows if r["speaker_affiliation"].strip() == aff]
        if not subset:
            continue
        n_true = sum(1 for r in subset if PRIMARY_MAP[label_of(r)])
        print(f"  {aff:<14}: {len(subset):>5} stmts, True rate = {n_true/len(subset)*100:5.1f}%")
    print()
    print("  Differing rates mean affiliation carries label signal. Whether the model")
    print("  should use it (genuine prior) or not (unfair shortcut) is a slide point.")

    # ------------------------------------------------------------------
    section("SUBJECTS / TOPICS")
    all_subjects = []
    for r in rows:
        for s in r["subjects"].split("$"):
            s = s.strip()
            if s:
                all_subjects.append(s)
    subj_counter = collections.Counter(all_subjects)
    print(f"  Unique subjects: {len(subj_counter)} | total tags: {len(all_subjects)} "
          f"| avg/statement: {len(all_subjects)/total:.1f}")
    print("\n  Top 15 subjects:")
    for subj, cnt in subj_counter.most_common(15):
        print(f"    {subj:<35} {cnt:>5}")

    # ------------------------------------------------------------------
    section("PERFORMANCE CEILING")
    print("  Verifying short factual claims requires external world knowledge and the")
    print("  human verdicts have limited inter-annotator agreement, so accuracy on this")
    print("  task is inherently bounded well below 100%. A calibrated result in the")
    print("  mid-to-high 0.60s is strong; effort is better spent on calibration,")
    print("  abstention, metadata ablation, and honest paired evaluation than on")
    print("  chasing accuracy past that ceiling.")

    # ------------------------------------------------------------------
    section("SUMMARY")
    print("  1. ~9.6k rows, 6-way ordinal truthfulness labels.")
    print("  2. PRIMARY binarization = middle split -> near-balanced; report sensitivity.")
    print("  3. Missing job/state/context ~1/3 of rows -> use explicit [unknown] tokens.")
    print("  4. Short statements (~18 words) -> cheap inference.")
    print("  5. Abstention in scope, framed via MODEL UNCERTAINTY (decline near p=0.5),")
    print("     NOT a large 'unjudgeable' population (only ~0.6% are literally too short).")
    print("  6. Heavy speaker concentration + non-person sources -> leakage/shortcuts;")
    print("     report speaker-disjoint vs random split gap as the headline leakage number.")
    print("  7. Accuracy is ceiling-bound -> a calibrated mid-to-high 0.60s is the target, not 100%.")
    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()

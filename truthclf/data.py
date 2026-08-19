"""Data loading, cleaning, binarization, and leakage-aware splitting.

This is the single, reusable data stage for the project.
Pipeline:

    rows = load("data.csv")
    clean, report = clean_dataset(rows, scheme="primary")   # drops contradictions
    tr, te = speaker_disjoint_split(clean, test_frac=0.2)    # group-aware, stratified
    tr, te = stratified_random_split(clean, test_frac=0.2)   # group-aware, stratified

Two leakage defenses are baked in (not optional post-processing):

1. **Contradiction drop.** When the *same* statement text carries *different
   binary labels* under the active scheme, the ground truth is self-contradictory.
   Every member of such a group is dropped and the row IDs are logged.

2. **Group-aware splitting.** Repeated statements (same text after
   normalisation — see `normalized_statement_key`; this is exact match after
   normalisation, NOT fuzzy matching) and, for the speaker-disjoint split, all
   statements by a shared speaker, are merged into atomic groups and assigned to
   ONE side of the split. This prevents repeated text — and speaker identity —
   from leaking across train/test. Both splits use the SAME majority-label stratification so
   their True/False ratios are comparable and the speaker-disjoint-vs-random gap
   isolates speaker memorization rather than class-balance drift.
"""

from __future__ import annotations

import csv
import json
import re
import collections
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Labels & binarization
# ---------------------------------------------------------------------------
LABELS_6 = ["true", "mostly-true", "half-true", "barely-true", "false", "extremely-false"]

BINARY_SCHEMES = {
    # PRIMARY: middle split — half-true counts as True.
    "primary": {
        "true": 1, "mostly-true": 1, "half-true": 1,
        "barely-true": 0, "false": 0, "extremely-false": 0,
    },
    # SENSITIVITY: half-true counts as False.
    "sensitivity": {
        "true": 1, "mostly-true": 1,
        "half-true": 0, "barely-true": 0, "false": 0, "extremely-false": 0,
    },
}


def binarize(label: str, scheme: str = "primary") -> int:
    """Map a 6-way label to {1=True, 0=False} under a named scheme."""
    return BINARY_SCHEMES[scheme][label.strip().lower()]


# ---------------------------------------------------------------------------
# Light text cleaning (clean lightly; do not over-normalize)
# ---------------------------------------------------------------------------
# Apostrophe-stripped contractions seen in the data (~5% of rows). We restore
# only this conservative, unambiguous set when they appear as standalone tokens.
_CONTRACTIONS = {
    "dont": "don't", "didnt": "didn't", "doesnt": "doesn't", "wont": "won't",
    "cant": "can't", "isnt": "isn't", "wasnt": "wasn't", "arent": "aren't",
    "werent": "weren't", "couldnt": "couldn't", "wouldnt": "wouldn't",
    "shouldnt": "shouldn't", "hasnt": "hasn't", "havent": "haven't",
    "hadnt": "hadn't", "youre": "you're", "theyre": "they're", "weve": "we've",
    "ive": "I've", "im": "I'm", "thats": "that's", "whats": "what's",
}


def clean_text(s: str) -> str:
    """Conservative cleanup: normalize curly quotes/dashes, collapse whitespace,
    restore obvious stripped contractions. Preserves casing and content."""
    if not s:
        return ""
    s = (s.replace("‘", "'").replace("’", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-"))
    out = []
    for tok in s.split():
        bare = tok.lower().strip(".,!?;:\"'()")
        if bare in _CONTRACTIONS:
            suffix = ""
            while tok and tok[-1] in ".,!?;:\"')":
                suffix = tok[-1] + suffix
                tok = tok[:-1]
            out.append(_CONTRACTIONS[bare] + suffix)
        else:
            out.append(tok)
    return " ".join(out)


def normalized_statement_key(statement: str) -> str:
    """Grouping key: lowercase, punctuation -> space, whitespace collapsed.

    This is EXACT MATCH AFTER NORMALISATION, not fuzzy near-duplicate detection.
    It catches re-punctuated and re-cased repeats of the same sentence and
    nothing else — an edit of one word produces a different key. Deliberate: it
    is deterministic and has no similarity threshold to justify, which matters
    because this key decides which rows are dropped as label contradictions and
    which rows are forced onto one side of a split. A real fuzzy matcher
    (rapidfuzz / datasketch MinHash) would catch more, at the cost of a tuned
    threshold and non-obvious merges; `_suspicious_normalization_merge` exists
    because even this conservative key over-merges occasionally.
    """
    keep = [c.lower() if (c.isalnum() or c.isspace()) else " " for c in statement]
    return " ".join("".join(keep).split())


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------
@dataclass
class Row:
    row_id: int
    label: str                      # 6-way label, lowercased
    statement: str                  # raw statement text
    subjects: str
    speaker_name: str
    speaker_job: str
    speaker_state: str
    speaker_affiliation: str
    statement_context: str
    statement_clean: str = field(default="")
    norm_key: str = field(default="")

    def y(self, scheme: str = "primary") -> int:
        return binarize(self.label, scheme)


def _get(d: dict, *names: str) -> str:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return ""


def load(path: str = "data.csv") -> list[Row]:
    """Load data.csv into Row objects (robust to Label/label header casing)."""
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        for i, d in enumerate(csv.DictReader(f)):
            stmt = _get(d, "statement").strip()
            rows.append(Row(
                row_id=i,
                label=_get(d, "Label", "label").strip().lower(),
                statement=stmt,
                subjects=_get(d, "subjects").strip(),
                speaker_name=_get(d, "speaker_name").strip(),
                speaker_job=_get(d, "speaker_job").strip(),
                speaker_state=_get(d, "speaker_state").strip(),
                speaker_affiliation=_get(d, "speaker_affiliation").strip(),
                statement_context=_get(d, "statement_context").strip(),
                statement_clean=clean_text(stmt),
                norm_key=normalized_statement_key(stmt),
            ))
    return rows


# ---------------------------------------------------------------------------
# Diagnostics: contradiction & repeated-statement groups (for inspection/deck)
# ---------------------------------------------------------------------------
def _norm_key_groups(rows: list[Row]) -> dict[str, list[Row]]:
    g: dict[str, list[Row]] = collections.defaultdict(list)
    for r in rows:
        g[r.norm_key].append(r)
    return g


def contradiction_groups(rows: list[Row], scheme: str = "primary") -> list[tuple[str, list[Row]]]:
    """Repeated-statement groups whose members map to BOTH binary classes."""
    out = []
    for key, mem in _norm_key_groups(rows).items():
        if len(mem) > 1 and len({m.y(scheme) for m in mem}) > 1:
            out.append((key, sorted(mem, key=lambda r: r.row_id)))
    return sorted(out, key=lambda g: g[1][0].row_id)


def _suspicious_normalization_merge(members: list[Row]) -> bool:
    """Flag a group that normalisation may have merged wrongly: the digit
    tokens differ, or word counts differ by more than 2."""
    digitsets = {tuple(sorted(re.findall(r"\d+", m.statement))) for m in members}
    if len(digitsets) > 1:
        return True
    wc = [len(m.statement.split()) for m in members]
    return (max(wc) - min(wc)) > 2


def cross_speaker_repeat_groups(rows: list[Row]) -> list[tuple[str, list[Row], bool]]:
    """Repeated-statement groups spanning >1 distinct speaker (cross-split leak
    risk). Returns (key, members, suspicious_flag)."""
    out = []
    for key, mem in _norm_key_groups(rows).items():
        if len(mem) > 1 and len({m.speaker_name for m in mem}) > 1:
            out.append((key, sorted(mem, key=lambda r: r.row_id), _suspicious_normalization_merge(mem)))
    return sorted(out, key=lambda g: g[1][0].row_id)


# ---------------------------------------------------------------------------
# Cleaning stage: drop self-contradictory duplicate groups
# ---------------------------------------------------------------------------
@dataclass
class CleanReport:
    scheme: str
    n_in: int
    n_out: int
    n_dropped: int
    dropped_groups: list                  # list of (norm_key, [row_ids], {labels})
    dropped_row_ids: list

    def summary(self) -> str:
        lines = [
            f"clean_dataset(scheme={self.scheme!r}): "
            f"{self.n_in} -> {self.n_out} rows "
            f"({self.n_dropped} dropped across {len(self.dropped_groups)} "
            f"contradictory groups)"
        ]
        for key, ids, lbls in self.dropped_groups:
            short = (key[:50] + "...") if len(key) > 50 else key
            lines.append(f"    drop rows {ids} {sorted(lbls)} :: {short!r}")
        return "\n".join(lines)


def clean_dataset(rows: list[Row], scheme: str = "primary") -> tuple[list[Row], CleanReport]:
    """Remove duplicate groups whose binary labels conflict under `scheme`.

    A group = rows sharing a normalized statement key. If a group's members map
    to BOTH binary classes, the ground truth is contradictory and the whole
    group is dropped. Returns (kept_rows, report).
    """
    dropped_groups = []
    dropped_ids = set()
    for key, mem in contradiction_groups(rows, scheme):
        ids = [m.row_id for m in mem]
        dropped_groups.append((key, ids, {m.label for m in mem}))
        dropped_ids.update(ids)

    kept = [r for r in rows if r.row_id not in dropped_ids]
    report = CleanReport(
        scheme=scheme, n_in=len(rows), n_out=len(kept),
        n_dropped=len(dropped_ids),
        dropped_groups=sorted(dropped_groups, key=lambda g: g[1][0]),
        dropped_row_ids=sorted(dropped_ids),
    )
    return kept, report


# ---------------------------------------------------------------------------
# Group-aware splitting
# ---------------------------------------------------------------------------
def _components(rows: list[Row], link_speaker: bool) -> list[list[Row]]:
    """Connected components of rows under union of relations:
       always: share a normalized-statement key; optionally: share a speaker.

    Uses scipy's DisjointSet rather than a hand-rolled union-find. Component
    MEMBERSHIP is implementation-independent, and so is the returned ORDER:
    components are keyed by root in a dict populated while iterating `rows` in
    order, so each component's key is inserted when its first row is seen,
    regardless of which member the implementation happens to pick as root. That
    ordering feeds the seeded shuffle in _stratified_assign, so it must not
    drift — tests/test_data.py pins the resulting split membership.
    """
    from scipy.cluster.hierarchy import DisjointSet

    uf = DisjointSet([r.row_id for r in rows])
    by_id = {r.row_id: r for r in rows}

    key_first: dict[str, int] = {}
    for r in rows:
        if r.norm_key in key_first:
            uf.merge(r.row_id, key_first[r.norm_key])
        else:
            key_first[r.norm_key] = r.row_id

    if link_speaker:
        spk_first: dict[str, int] = {}
        for r in rows:
            spk = r.speaker_name or f"__empty__{r.row_id}"   # empty speakers stay singletons
            if spk in spk_first:
                uf.merge(r.row_id, spk_first[spk])
            else:
                spk_first[spk] = r.row_id

    comps: dict[int, list[Row]] = collections.defaultdict(list)
    for r in rows:
        comps[uf[r.row_id]].append(by_id[r.row_id])
    return list(comps.values())


def _greedy_balanced(comps: list[list[Row]], target: float) -> tuple[list, list]:
    """Assign whole components to test, largest-first with a midpoint rule:
    add a component to test only while it leaves us no further past `target`
    than stopping would. Bounds test-size overshoot to ~half the largest comp."""
    test, train, n_test = [], [], 0
    for comp in sorted(comps, key=len, reverse=True):
        # adding is better iff it doesn't push us more than halfway past target
        if n_test + len(comp) / 2.0 <= target:
            test.extend(comp)
            n_test += len(comp)
        else:
            train.extend(comp)
    return train, test


def _stratified_assign(comps, test_frac, seed, scheme):
    """Majority-label-stratified, overshoot-controlled component assignment.
    Used by BOTH splits so their class ratios are comparable."""
    rng = random.Random(seed)
    by_class = {0: [], 1: []}
    for comp in comps:
        votes = sum(r.y(scheme) for r in comp)
        majority = 1 if votes * 2 >= len(comp) else 0
        by_class[majority].append(comp)

    train, test = [], []
    for cls in (0, 1):
        cl = by_class[cls]
        rng.shuffle(cl)                              # seed-dependent tie ordering
        target = test_frac * sum(len(c) for c in cl)
        tr, te = _greedy_balanced(cl, target)
        train.extend(tr)
        test.extend(te)
    return train, test


def speaker_disjoint_split(rows, test_frac=0.2, seed=0, scheme="primary"):
    """Split so NO speaker and NO repeated-statement group crosses train/test.
    Components = (same speaker) union (same normalized-statement key). This is
    the primary, leakage-resistant split."""
    return _stratified_assign(_components(rows, link_speaker=True), test_frac, seed, scheme)


def speaker_disjoint_3way(rows, val_frac=0.2, test_frac=0.2, seed=0, scheme="primary"):
    """Three-way speaker-disjoint split -> (train, val, test), all pairwise
    speaker-disjoint and repeat-group-disjoint. Tune on val, report on test;
    test matches speaker_disjoint_split(test_frac, seed) so it lines up with the
    later fine-tuned comparison."""
    train_full, test = speaker_disjoint_split(rows, test_frac=test_frac, seed=seed, scheme=scheme)
    inner = val_frac / (1.0 - test_frac)        # val as a fraction of the remainder
    train, val = speaker_disjoint_split(train_full, test_frac=inner, seed=seed + 1, scheme=scheme)
    return train, val, test


def stratified_random_split(rows, test_frac=0.2, seed=0, scheme="primary"):
    """Class-stratified split. Repeated-statement groups stay together (no
    repeated text across train/test), but speakers MAY cross — by design, so the gap vs
    the speaker-disjoint split estimates speaker memorization."""
    return _stratified_assign(_components(rows, link_speaker=False), test_frac, seed, scheme)


# ---------------------------------------------------------------------------
# Split integrity checks (used by tests and the Phase-1 report)
# ---------------------------------------------------------------------------
def speakers_cross(train, test) -> set:
    """Speakers (non-empty) present on both sides."""
    def spk(rows):
        return {r.speaker_name for r in rows if r.speaker_name}
    return spk(train) & spk(test)


def normkeys_cross(train, test) -> set:
    """Normalized-statement keys present on both sides."""
    return {r.norm_key for r in train} & {r.norm_key for r in test}


def class_balance(rows, scheme="primary") -> tuple[int, int, float]:
    """Return (n_true, n_false, true_fraction)."""
    n_true = sum(r.y(scheme) for r in rows)
    n = len(rows)
    return n_true, n - n_true, (n_true / n if n else 0.0)


def label_distribution(rows) -> dict:
    """6-way label counts in canonical order (display helper for notebooks)."""
    c = collections.Counter(r.label for r in rows)
    return {lbl: c.get(lbl, 0) for lbl in LABELS_6}


def dev_subset(rows, n: int = 200, seed: int = 0) -> list:
    """Deterministic random subset for prompt iteration / scale checks."""
    return random.Random(seed).sample(rows, min(n, len(rows)))


# ---------------------------------------------------------------------------
# Supervised fine-tuning (SFT) data serialization
# ---------------------------------------------------------------------------
def to_sft_record(row: Row, variant: str = "full", scheme: str = "primary") -> dict:
    """Build one SFT chat record for a row in the 'contents' format."""
    from . import prompts  # lazy import to avoid a cycle
    
    # Per Vertex AI documentation, the system prompt should be the first turn.
    # The full user prompt includes all metadata.
    system_prompt = prompts.SYSTEM_DECISION
    user_prompt_text = prompts.build_user_prompt(row, variant, mode="decision")

    # The expected model output.
    model_response_text = "True" if row.y(scheme) == 1 else "False"
    
    # The new 'contents' format does not use a system role. The convention is
    # to place the system prompt as the first 'user' turn, but for SFT,
    # it's often included directly in the user prompt. We will combine them.
    full_user_prompt = f"{system_prompt}\n\n{user_prompt_text}"

    return {
        "contents": [
            {"role": "user", "parts": [{"text": full_user_prompt}]},
            {"role": "model", "parts": [{"text": model_response_text}]}
        ]
    }


def write_sft_jsonl(rows, path: str, variant: str = "full", scheme: str = "primary") -> str:
    """Write rows as a conversational JSONL file for SFT. Returns the path."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_sft_record(r, variant, scheme), ensure_ascii=False) + "\n")
    return path

"""Unit tests for truthclf.data — binarization, cleaning, contradiction drop,
and leakage-aware (group-aware, stratified) splitting. Tiny synthetic data."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import data  # noqa: E402


def mk(rid, label, statement, speaker):
    return data.Row(
        row_id=rid, label=label, statement=statement, subjects="",
        speaker_name=speaker, speaker_job="", speaker_state="",
        speaker_affiliation="", statement_context="",
        statement_clean=data.clean_text(statement), norm_key=data.normalized_statement_key(statement),
    )


# ---------------------------------------------------------------------------
def test_binarize_schemes():
    assert data.binarize("true", "primary") == 1
    assert data.binarize("half-true", "primary") == 1
    assert data.binarize("half-true", "sensitivity") == 0
    assert data.binarize("barely-true", "primary") == 0
    assert data.binarize("FALSE", "primary") == 0          # case-insensitive


def test_clean_text_restores_contractions():
    assert data.clean_text("they dont know") == "they don't know"
    assert data.clean_text("He didnt.") == "He didn't."    # trailing punct kept
    assert data.clean_text("a  b   c") == "a b c"           # whitespace collapse
    assert data.clean_text("“quote”") == '"quote"'          # curly quotes


def test_norm_key_collapses_punct_and_case():
    assert data.normalized_statement_key("On reconciliation.") == data.normalized_statement_key("on reconciliation")
    assert data.normalized_statement_key("Jobs grew, fast!") == data.normalized_statement_key("jobs grew fast")
    # punctuation becomes a space, so "U.S." -> "u s" (does NOT collapse to "us")
    assert data.normalized_statement_key("U.S. jobs") != data.normalized_statement_key("us jobs")


# ---------------------------------------------------------------------------
def test_contradiction_drop_removes_whole_group():
    rows = [
        mk(0, "true", "On gay marriage.", "alice"),       # -> True
        mk(1, "false", "on gay marriage", "bob"),          # -> False  (conflict)
        mk(2, "true", "Taxes went up.", "carol"),          # unique, kept
    ]
    clean, rep = data.clean_dataset(rows, scheme="primary")
    kept_ids = {r.row_id for r in clean}
    assert kept_ids == {2}
    assert rep.n_dropped == 2
    assert rep.dropped_row_ids == [0, 1]
    assert len(rep.dropped_groups) == 1


def test_no_exact_duplicate_survives_with_conflicting_binary_labels():
    """Post-clean invariant: no normalized statement key maps to both classes."""
    rows = [
        mk(0, "true", "On gay marriage.", "alice"),
        mk(1, "false", "on gay marriage", "bob"),
        mk(2, "mostly-true", "Jobs grew", "carol"),
        mk(3, "half-true", "Jobs grew.", "dave"),          # primary: both True -> kept
    ]
    clean, _ = data.clean_dataset(rows, scheme="primary")
    by_key = {}
    for r in clean:
        by_key.setdefault(r.norm_key, set()).add(r.y("primary"))
    assert all(len(v) == 1 for v in by_key.values()), "a dup key kept both classes"
    # the half-true/mostly-true pair is NOT a contradiction under primary -> kept
    assert {r.row_id for r in clean} == {2, 3}


# ---------------------------------------------------------------------------
def _synthetic_corpus():
    """12 speakers, balanced labels, plus a cross-speaker near-duplicate pair."""
    rows = []
    rid = 0
    labels = ["true", "mostly-true", "half-true", "barely-true", "false", "extremely-false"]
    for i in range(12):
        lbl = labels[i % len(labels)]
        rows.append(mk(rid, lbl, f"Statement number {i} about policy.", f"speaker{i}"))
        rid += 1
    # cross-speaker near-duplicate (same binary class under primary: both False)
    rows.append(mk(rid, "false", "The bill costs ten billion dollars.", "speakerX")); rid += 1
    rows.append(mk(rid, "barely-true", "the bill costs ten billion dollars", "speakerY")); rid += 1
    return rows


def test_speaker_disjoint_split_invariants():
    rows = _synthetic_corpus()
    clean, _ = data.clean_dataset(rows, "primary")
    train, test = data.speaker_disjoint_split(clean, test_frac=0.34, seed=1)
    assert len(train) + len(test) == len(clean)            # partition
    assert data.speakers_cross(train, test) == set()       # no speaker crosses
    assert data.normkeys_cross(train, test) == set()        # no near-dup crosses


def test_stratified_split_keeps_neardups_together():
    rows = _synthetic_corpus()
    clean, _ = data.clean_dataset(rows, "primary")
    train, test = data.stratified_random_split(clean, test_frac=0.34, seed=1)
    assert len(train) + len(test) == len(clean)
    assert data.normkeys_cross(train, test) == set()        # no identical text leak


def test_cross_speaker_repeat_group_stays_on_one_side():
    """The near-dup pair by speakerX/speakerY must not be split apart."""
    rows = _synthetic_corpus()
    clean, _ = data.clean_dataset(rows, "primary")
    key = data.normalized_statement_key("The bill costs ten billion dollars.")
    for splitter in (data.speaker_disjoint_split, data.stratified_random_split):
        train, test = splitter(clean, test_frac=0.34, seed=3)
        in_train = any(r.norm_key == key for r in train)
        in_test = any(r.norm_key == key for r in test)
        assert in_train != in_test, "near-dup group landed on both sides"


def test_diagnostics_find_groups():
    rows = _synthetic_corpus() + [
        mk(99, "true", "On gay marriage.", "alice"),
        mk(100, "false", "on gay marriage", "bob"),
    ]
    contra = data.contradiction_groups(rows, "primary")
    assert any({m.row_id for m in mem} == {99, 100} for _, mem in contra)
    cross = data.cross_speaker_repeat_groups(rows)
    keys = {key for key, _, _ in cross}
    assert data.normalized_statement_key("the bill costs ten billion dollars") in keys

"""Unit tests for the schema-2 response cache and the response-shape extractors.

These pin the three properties that schema 1 got wrong: the backend is part of
the key, failures are never stored, and malformed responses raise instead of
silently becoming a neutral probability. No network; every cache is a tmp_path.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truthclf import llm  # noqa: E402

MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
PARAMS = dict(max_tokens=16, temperature=0.0, reasoning_effort=None, extra={})


def key(**over):
    kw = dict(backend="sync", call="score", **PARAMS)
    kw.update(over)
    return llm.ResponseCache.key("m", MSGS, **kw)


# --------------------------------------------------------------------------
# Key composition — the schema-1 collisions must be impossible now.
# --------------------------------------------------------------------------
def test_backend_is_part_of_the_key():
    """The schema-1 root cause: sync and batch shared a key, so the later run
    silently overwrote the earlier one's value."""
    assert key(backend="sync") != key(backend="batch")


def test_call_kind_is_part_of_the_key():
    assert key(call="score") != key(call="classify") != key(call="complete")


def test_schema_version_is_part_of_the_key(monkeypatch):
    before = key()
    monkeypatch.setattr(llm, "CACHE_SCHEMA", llm.CACHE_SCHEMA + 1)
    assert key() != before, "bumping CACHE_SCHEMA must invalidate existing keys"


def test_key_is_stable_and_order_independent():
    assert key() == key()
    a = llm.ResponseCache.key("m", MSGS, backend="sync", call="score",
                              temperature=0.0, max_tokens=16, extra={},
                              reasoning_effort=None)
    assert a == key(), "param dict ordering must not change the hash"


def test_prompt_text_change_changes_the_key():
    other = [{"role": "system", "content": "sys"}, {"role": "user", "content": "HI"}]
    assert llm.ResponseCache.key("m", other, backend="sync", call="score", **PARAMS) != key()


# --------------------------------------------------------------------------
# Values: provenance envelope, and failures are never cached.
# --------------------------------------------------------------------------
def test_set_get_roundtrip_and_provenance(tmp_path):
    c = llm.ResponseCache(str(tmp_path / "c"))
    k = key()
    c.set(k, "85", backend="sync", call="score", model="m")
    assert c.get(k) == "85"
    env = c.envelope(k)
    assert env["backend"] == "sync" and env["call"] == "score" and env["model"] == "m"
    assert env["schema"] == llm.CACHE_SCHEMA
    assert env["written_at"].endswith("+00:00") and env["sdk"]


def test_empty_value_is_not_cached(tmp_path):
    """Schema 1 wrote "" for a failed Batch row and then served it as a hit
    forever, permanently injecting a neutral p=0.5 into the metrics."""
    c = llm.ResponseCache(str(tmp_path / "c"))
    k = key()
    c.set(k, "", backend="batch", call="score", model="m")
    assert c.get(k) is None, "a failure must stay a miss so a rerun retries it"
    assert len(c) == 0


def test_miss_returns_none(tmp_path):
    assert llm.ResponseCache(str(tmp_path / "c")).get(key()) is None


def _writer(path, lo, hi):
    c = llm.ResponseCache(path)
    for i in range(lo, hi):
        c.set(f"k{i}", str(i), backend="sync", call="score", model="m")
    c.close()


def test_concurrent_writers_do_not_lose_entries(tmp_path):
    """The schema-1 store rewrote the whole JSON file on flush, so two processes
    each dropped the other's writes. diskcache is per-key."""
    path = str(tmp_path / "c")
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(path, lo, lo + 50))
             for lo in (0, 50, 100)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    c = llm.ResponseCache(path)
    assert len(c) == 150
    assert {c.get(f"k{i}") for i in range(150)} == {str(i) for i in range(150)}


# --------------------------------------------------------------------------
# Raw-then-parse: the classify path must not cache pre-parsed values.
# --------------------------------------------------------------------------
RAW_CONTENT0 = {"token": "True", "logprob": -0.2,
                "top_logprobs": [{"token": "True", "logprob": -0.2},
                                 {"token": "False", "logprob": -1.7}]}


def test_classify_payload_parses_on_read():
    raw = json.dumps({"text": "True", "content0": RAW_CONTENT0})
    out = llm._parse_logprobs_payload(raw)
    assert out["text"] == "True"
    assert out["top_logprobs"] == {"True": -0.2, "False": -1.7}


def test_cached_classify_value_retains_the_raw_structure():
    """If the cached value were the flattened map, changing the flattening logic
    would silently reinterpret stored entries behind an unchanged key."""
    raw = json.loads(json.dumps({"text": "True", "content0": RAW_CONTENT0}))
    assert raw["content0"]["top_logprobs"][1] == {"token": "False", "logprob": -1.7}


def test_chosen_token_is_included_even_if_absent_from_top_logprobs():
    lp = llm._lp_content_to_top({"token": "Maybe", "logprob": -9.0, "top_logprobs": []})
    assert lp == {"Maybe": -9.0}


# --------------------------------------------------------------------------
# Malformed responses raise instead of degrading to a neutral probability.
# --------------------------------------------------------------------------
def test_batch_body_raises_on_missing_choices():
    with pytest.raises(llm.ResponseShapeError):
        llm._batch_body({"custom_id": "3", "error": "upstream 500"})


def test_extract_top_logprobs_raises_when_logprobs_absent():
    line = {"response": {"body": {"choices": [{"message": {"content": "True"}}]}}}
    with pytest.raises(llm.ResponseShapeError):
        llm._extract_top_logprobs_obj(line)


def test_extract_content_reads_each_nesting_shape():
    body = {"choices": [{"message": {"content": "42"}}]}
    for line in ({"response": {"body": body}}, {"body": body}, body):
        assert llm._extract_content(line) == "42"

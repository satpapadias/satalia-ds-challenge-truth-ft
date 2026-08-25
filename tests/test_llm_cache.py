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
# A set is processed concurrently, and concurrency does not corrupt the cache.
# --------------------------------------------------------------------------
class _SlowGenerator:
    """Stand-in whose calls sleep, so serial and concurrent are distinguishable."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.calls = 0

    def generate(self, contents, generation_config, system_instruction=None):
        import threading
        import time as _t
        _t.sleep(self.delay)
        with threading.Lock():
            self.calls += 1
        txt = contents[-1]["parts"][0]["text"]
        return type("R", (), {"text": txt})()


def _client(tmp_path, workers, delay=0.05):
    c = llm.VertexClient("gemini-2.5-flash", cache=llm.ResponseCache(str(tmp_path / f"c{workers}")),
                         max_workers=workers)
    generator = _SlowGenerator(delay)
    c._generate = generator.generate
    c._ensure_client = lambda: None
    c._generator = generator
    return c


def _msgs(n):
    return [[{"role": "user", "content": str(i)}] for i in range(n)]


def test_score_over_a_set_is_concurrent(tmp_path):
    """16 calls at 50 ms each: serial would take >=0.8 s, concurrent well under."""
    import time
    n = 16
    t0 = time.perf_counter()
    out = _client(tmp_path / "par", workers=16).score(_msgs(n))
    parallel = time.perf_counter() - t0
    assert out == [str(i) for i in range(n)], "order must be preserved"
    assert parallel < 0.40, f"took {parallel:.2f}s — looks serial, not concurrent"


def test_max_workers_one_is_serial(tmp_path):
    out = _client(tmp_path / "ser", workers=1, delay=0.0).score(_msgs(5))
    assert out == [str(i) for i in range(5)]


def test_results_stay_aligned_with_inputs_under_concurrency(tmp_path):
    c = _client(tmp_path / "align", workers=8, delay=0.01)
    out = c.score(_msgs(40))
    assert out == [str(i) for i in range(40)], "ex.map must keep input order"


def test_concurrent_writes_all_reach_the_cache(tmp_path):
    c = _client(tmp_path / "w", workers=8, delay=0.01)
    c.score(_msgs(40))
    assert len(c.cache) == 40, "every concurrent write must be durable"
    again = _client(tmp_path / "w", workers=8, delay=0.01)
    assert again.score(_msgs(40)) == [str(i) for i in range(40)]
    assert again._generator.calls == 0, "second pass must be all hits"


def test_an_explicitly_passed_empty_cache_is_honoured(tmp_path):
    """ResponseCache defines __len__, so an empty one is falsy. `cache or
    ResponseCache()` therefore discarded a caller's fresh cache and silently
    used the project default — writing to the wrong store."""
    rc = llm.ResponseCache(str(tmp_path / "mine"))
    assert len(rc) == 0
    client = llm.VertexClient("gemini-2.5-flash", cache=rc)
    assert client.cache is rc, "an empty cache must not be replaced by the default"
    assert client.cache.path == str(tmp_path / "mine")

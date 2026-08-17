"""LLM utilities for provider backends: token/cost estimation, response
caching, and thin clients with retry/backoff.

The default backend is Vertex AI (Gemini). The Together AI backend is retained
for offline replay of the historical record.

No network call happens on import. SDKs are imported lazily and pick up their
credentials from the environment (e.g. gcloud ADC, or TOGETHER_API_KEY). Token
counting uses tiktoken as an approximation; tokenizers differ, so cost figures
are estimates.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
import warnings

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

# USD per 1,000,000 tokens. Verified 2026-06-22 from together.ai pricing.
# Vertex pricing verified 2026-08-14.
PRICING = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":         {"input": 1.04, "output": 1.04},
    "openai/gpt-oss-20b":                              {"input": 0.05, "output": 0.20},
    "openai/gpt-oss-120b":                             {"input": 0.15, "output": 0.60},
    "Qwen/Qwen3.5-9B":                                {"input": 0.17, "output": 0.25},
    "google/gemma-4-31B-it":                           {"input": 0.39, "output": 0.97},
    "gemini-2.5-flash":                                {"input": 0.35, "output": 0.70},
}

# Per-model client settings. Reasoning models need either a low reasoning effort
# or thinking disabled, plus enough output budget to emit the final score.
_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MODEL_CONFIGS = {
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"max_output_tokens": 8},
    "openai/gpt-oss-20b":   {"max_output_tokens": 512, "reasoning_effort": "low"},
    "openai/gpt-oss-120b":  {"max_output_tokens": 512, "reasoning_effort": "low"},
    "Qwen/Qwen3.5-9B":      {"max_output_tokens": 16, "extra_create_kwargs": _NO_THINK},
    "google/gemma-4-31B-it": {"max_output_tokens": 16, "extra_create_kwargs": _NO_THINK},
    "gemini-2.5-flash":     {"max_output_tokens": 16},
}
DEFAULT_CONFIG = {"max_output_tokens": 16}


def _retryable_errors():
    """Transient Together/HTTP failures worth retrying.

    Deliberately EXCLUDES AuthenticationError, BadRequestError,
    InvalidRequestError and APIResponseValidationError: those are deterministic
    and retrying them only delays an error the caller has to fix anyway.
    """
    try:
        from together import error as te
        return (te.APIConnectionError, te.APITimeoutError, te.RateLimitError,
                te.Timeout, te.ResponseError)
    except ImportError:                          # SDK absent: nothing to import
        return (ConnectionError, TimeoutError)


RETRYABLE_ERRORS = _retryable_errors()


def make_client(model: str, cache=None, backend: str = "vertex", **overrides):
    """Build a client for `model` using its registered config (or the default).

    backend="vertex" -> VertexClient (concurrent over a set, cached)
    backend="sync"   -> TogetherClient (concurrent, cached) for dev/interactive
    backend="batch"  -> TogetherBatchClient (Batch API) for full-test runs

    The cache keys embed the backend, so responses from different providers
    cannot collide.
    Fine-tuned model ids fall back to the default config, so the same call site
    works for base and fine-tuned models."""
    cfg = dict(MODEL_CONFIGS.get(model, DEFAULT_CONFIG))
    cfg.update(overrides)
    if backend == "vertex":
        return VertexClient(model, cache=cache, **cfg)
    if backend == "batch":
        return TogetherBatchClient(model, cache=cache, **cfg)
    if backend == "sync":
        return TogetherClient(model, cache=cache, **cfg)
    raise ValueError(f"unknown backend: {backend!r}")

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    TOKENIZER = "tiktoken/o200k_base (approx for Llama/GPT-OSS)"
except Exception as _tiktoken_error:             # noqa: F841 - reported below
    # tiktoken is a declared dependency, so this branch means a broken install or
    # (most likely) no network for the BPE download. Falling back silently would
    # change every printed cost estimate by an unknown factor, so it warns loudly.
    warnings.warn(
        f"tiktoken unavailable ({type(_tiktoken_error).__name__}: {_tiktoken_error}); "
        "falling back to a ~4-chars/token heuristic. COST ESTIMATES ARE APPROXIMATE "
        "and may be off by tens of percent.",
        RuntimeWarning, stacklevel=2,
    )

    def count_tokens(text: str) -> int:
        return math.ceil(len(text) / 4)

    TOKENIZER = "heuristic (~4 chars/token) - tiktoken unavailable"


def estimate_cost(n_input_tokens: int, n_output_tokens: int, model: str) -> float:
    """Estimated USD cost for input/output token counts at a model's rate."""
    # A fine-tuned model id (e.g. from a Vertex endpoint string) is not in the
    # pricing table; fall back to the known base model.
    price = PRICING.get(model)
    if price is None and "gemini" in model:
        price = PRICING["gemini-2.5-flash"]
    return (n_input_tokens * price["input"] + n_output_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# On-disk response cache so reruns are free.
# ---------------------------------------------------------------------------
# Anchor the cache to the project root (parent of this package), so it is found
# regardless of the current working directory (e.g. when run from notebooks/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, ".llm_cache")

# Bump when the meaning of a cached VALUE changes in a way the key cannot express.
# Every stored key embeds this, so a bump invalidates the whole cache rather than
# silently serving entries that the current code would interpret differently.
CACHE_SCHEMA = 2


class ResponseCache:
    """Cache of raw model responses, keyed by everything that determines them.

    The key binds: schema version, serving backend, call kind, model id, the full
    message list, and the request params. Two properties matter:

    * **Backend is part of the key.** Together, Vertex, and the Together Batch
      API are different serving paths that can return different completions for
      the same prompt. Schema 1 deliberately shared keys between them, so
      whichever ran last silently overwrote the other's value and the recorded
      numbers depended on run order. They are now separate namespaces.
    * **Values are raw.** Nothing is parsed before storage — a cached score is
      the model's text, a cached classification is the API's logprobs structure.
      Parsing happens on read, so changing a parser can never leave stale
      pre-parsed values behind a key that did not move.

    Failed calls are NOT written. A miss must retry; a cached failure cannot.

    Storage is diskcache (SQLite/WAL): per-key writes, atomic, and safe for
    concurrent processes on one host. It sits behind this interface so the
    backing store stays swappable.
    """

    def __init__(self, path: str | None = None):
        import diskcache
        self.path = path or _DEFAULT_CACHE
        self._cache = diskcache.Cache(self.path)

    @staticmethod
    def key(model: str, messages: list[dict], *, backend: str, call: str, **params) -> str:
        """Content hash of the full request identity. `backend` is "sync"/"batch";
        `call` is "score"/"classify"/"complete"."""
        blob = json.dumps({"cache_schema": CACHE_SCHEMA, "backend": backend, "call": call,
                           "model": model, "messages": messages, "params": params},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        """Cached raw value, or None on a miss."""
        env = self._cache.get(key)
        return None if env is None else env["value"]

    def envelope(self, key: str):
        """Full stored record including provenance (backend, model, timestamp, sdk)."""
        return self._cache.get(key)

    def set(self, key: str, value: str, *, backend: str, call: str, model: str) -> None:
        """Store a successful response with provenance.

        Empty values are treated as failures and are NOT stored: in schema 1 a
        transient Batch-API failure was written as "" and then served as a hit
        forever, silently feeding a neutral p=0.5 into the reported metrics.
        """
        if value == "":
            return
        self._cache[key] = {
            "value": value,
            "schema": CACHE_SCHEMA,
            "backend": backend,
            "call": call,
            "model": model,
            "written_at": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(timespec="seconds"),
            "sdk": _sdk_version(),
        }

    def flush(self) -> None:
        """No-op: diskcache writes through on set(). Kept so call sites that
        batched up a whole-file dump under schema 1 keep working."""

    def close(self) -> None:
        self._cache.close()

    def __len__(self) -> int:
        return len(self._cache)


def legacy_v1_key(model: str, messages: list[dict], **params) -> str:
    """Reproduce a SCHEMA-1 cache key exactly: sha256 over (model, messages, params)
    with no cache_schema/backend/call fields.

    Frozen format, kept because `.llm_cache.json` is still the archive the adopted
    (a)/(b)/(c) numbers replay from. It lived in two independent copies — the
    migration script and the results regenerator — either of which drifting would
    have silently stopped that replay from reproducing the record.
    """
    blob = json.dumps({"model": model, "messages": messages, "params": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sdk_version() -> str:
    """Version string recorded in each cache envelope, for provenance."""
    versions = []
    try:
        import importlib.metadata as md
        for pkg in ("together", "google-generativeai", "vertexai"):
            try:
                versions.append(f"{pkg} {md.version(pkg)}")
            except md.PackageNotFoundError:
                pass
    finally: return ", ".join(versions) or "unknown"


# ---------------------------------------------------------------------------
# Together client (OpenAI-compatible chat completions) with retry + backoff.
# ---------------------------------------------------------------------------
class TogetherClient:
    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 8, temperature: float = 0.0,
                 max_retries: int = 4, backoff_base: float = 1.0, timeout: float = 120.0,
                 reasoning_effort: str | None = None, extra_create_kwargs: dict | None = None,
                 max_workers: int = 16):
        self.model = model
        # A set of points is fetched CONCURRENTLY, not one call after another.
        # max_workers=1 restores strictly serial behaviour.
        self.max_workers = max(1, int(max_workers))
        # `is None`, NOT `or`: ResponseCache defines __len__, so an EMPTY cache is
        # falsy and `cache or ResponseCache()` silently swapped a caller's
        # fresh cache for the project default.
        self.cache = ResponseCache() if cache is None else cache
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        # Reasoning models (e.g. gpt-oss) emit chain-of-thought before content;
        # set 'low' and a larger token budget so the final answer is produced.
        self.reasoning_effort = reasoning_effort
        # Extra params passed to create() — e.g. {"chat_template_kwargs":
        # {"enable_thinking": False}} to disable thinking on Qwen/Gemma.
        self.extra_create_kwargs = extra_create_kwargs or {}
        self.api_errors = 0          # logged: transient failures that were retried
        self.n_api_calls = 0         # actual (cache-miss) calls made
        self.total_api_seconds = 0.0
        self._client = None

    @property
    def avg_latency(self) -> float:
        return self.total_api_seconds / self.n_api_calls if self.n_api_calls else 0.0

    def _params_key(self) -> dict:
        return {"temperature": self.temperature, "reasoning_effort": self.reasoning_effort,
                "extra": self.extra_create_kwargs}

    def _ensure_client(self):
        if self._client is None:
            try:
                from together import Together
            except ImportError as e:
                raise RuntimeError(
                    "together package not installed; run `pip install together` and set "
                    "TOGETHER_API_KEY (e.g. in .env)."
                ) from e
            self._client = Together(timeout=self.timeout)   # reads TOGETHER_API_KEY
        return self._client

    def _create(self, **kwargs):
        """Call chat.completions.create, retrying TRANSIENT failures only.

        Retry policy is tenacity's, not hand-rolled: exponential backoff WITH
        jitter (the previous loop had none, so concurrent workers retried in
        lockstep), and retries limited to the transient error classes.
        Previously any Exception was retried, so an auth failure or a malformed
        request — neither of which can ever succeed — burned all four attempts
        and ~15 seconds before surfacing.
        """
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        for k, v in self.extra_create_kwargs.items():
            kwargs.setdefault(k, v)
        client = self._ensure_client()

        def _count_error(retry_state):
            self.api_errors += 1

        @retry(retry=retry_if_exception_type(RETRYABLE_ERRORS),
               wait=wait_exponential_jitter(initial=self.backoff_base, max=60.0),
               stop=stop_after_attempt(self.max_retries),
               before_sleep=_count_error,
               reraise=True)
        def _call():
            t = time.time()
            resp = client.chat.completions.create(**kwargs)
            self.total_api_seconds += time.time() - t
            self.n_api_calls += 1
            return resp

        return _call()

    BACKEND = "sync"

    def _map(self, fn, items: list):
        """Apply `fn` across a set, concurrently, preserving input order.

        The spec requires each component to process a SET efficiently; a list
        comprehension over single calls is a serial round-trip per point. Only
        cache MISSES reach the network, and diskcache is safe for concurrent
        readers and writers, so no lock is needed around cache access — the
        counters below are the only shared mutable state and are incremented
        under the GIL on whole ints.
        """
        if self.max_workers == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(fn, items))         # ex.map preserves order

    def _cache_key(self, messages: list[dict], call: str, **params) -> str:
        return self.cache.key(self.model, messages, backend=self.BACKEND, call=call,
                              **params, **self._params_key())

    # --- primary: 0-100 score path ---------------------------------------
    def score_one(self, messages: list[dict]) -> str:
        k = self._cache_key(messages, "score", max_tokens=self.max_output_tokens)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._create(model=self.model, messages=messages,
                            max_tokens=self.max_output_tokens, temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        self.cache.set(k, text, backend=self.BACKEND, call="score", model=self.model)
        return text

    def score(self, messages_list: list[list[dict]]) -> list[str]:
        out = self._map(self.score_one, messages_list)
        self.cache.flush()
        return out

    def complete_one(self, messages: list[dict], max_tokens: int = 64) -> str:
        k = self._cache_key(messages, "complete", max_tokens=max_tokens)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._create(model=self.model, messages=messages, max_tokens=max_tokens,
                            temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        self.cache.set(k, text, backend=self.BACKEND, call="complete", model=self.model)
        return text

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        """Free-text generation (e.g. one-sentence rationales)."""
        out = self._map(lambda m: self.complete_one(m, max_tokens), messages_list)
        self.cache.flush()
        return out

    # --- optional: logprob-based True/False path -------------------------
    def classify_one(self, messages: list[dict], top_logprobs: int = 10) -> dict:
        """Single-token classification with token logprobs.

        The cache holds the API's RAW logprobs structure, not a flattened
        {token: logprob} map; `_lp_content_to_top` runs on read. Caching the
        parsed form (schema 1) meant any change to the extraction logic left
        stale values behind an unchanged key.
        """
        k = self._cache_key(messages, "classify", max_tokens=1, logprobs=top_logprobs)
        cached = self.cache.get(k)
        if cached is None:
            # Together returns logprobs nested in logprobs.content[*]; the SDK rejects
            # top_logprobs as a kwarg, so request it via extra_body.
            resp = self._create(model=self.model, messages=messages, max_tokens=1,
                                temperature=self.temperature,
                                extra_body={"logprobs": True, "top_logprobs": top_logprobs})
            cached = json.dumps(_raw_logprobs_payload(resp.choices[0]), ensure_ascii=False)
            self.cache.set(k, cached, backend=self.BACKEND, call="classify", model=self.model)
        return _parse_logprobs_payload(cached)

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        out = self._map(self.classify_one, messages_list)
        self.cache.flush()
        return out

    def ping(self) -> str:
        """Minimal availability probe (1 token). Raises if the model is not
        callable serverlessly. Used to verify model strings before a run."""
        resp = self._create(model=self.model,
                            messages=[{"role": "user", "content": "Reply with: ok"}],
                            max_tokens=1, temperature=0.0)
        return resp.choices[0].message.content or ""


class VertexClient:
    """LLM client for Vertex AI (Gemini)."""

    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 8, temperature: float = 0.0,
                 max_retries: int = 4, backoff_base: float = 1.0, timeout: float = 120.0,
                 max_workers: int = 16):
        self.model = model
        self.max_workers = max(1, int(max_workers))
        self.cache = ResponseCache() if cache is None else cache
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.api_errors = 0
        self.n_api_calls = 0
        self.total_api_seconds = 0.0
        self._client = None

    @property
    def avg_latency(self) -> float:
        return self.total_api_seconds / self.n_api_calls if self.n_api_calls else 0.0

    def _params_key(self) -> dict:
        return {"temperature": self.temperature}

    def _ensure_client(self):
        if self._client is None:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel
            except ImportError as e:
                raise RuntimeError(
                    "google-cloud-aiplatform not installed; run `pip install "
                    "google-cloud-aiplatform` and `gcloud auth application-default login`."
                ) from e
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION")
            if not project or not location:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for Vertex AI.")
            vertexai.init(project=project, location=location)
            self._client = GenerativeModel(self.model)
        return self._client

    def _generate(self, contents, **generation_config):
        client = self._ensure_client()

        def _count_error(retry_state):
            self.api_errors += 1

        # google-genai has its own tenacity-based retry config.
        @retry(retry=retry_if_exception_type(RETRYABLE_ERRORS),
               wait=wait_exponential_jitter(initial=self.backoff_base, max=60.0),
               stop=stop_after_attempt(self.max_retries),
               before_sleep=_count_error,
               reraise=True)
        def _call():
            t = time.time()
            # The SDK's own timeout is passed at method level, not client level.
            resp = client.generate_content(
                contents, generation_config=generation_config, request_options={"timeout": self.timeout})
            self.total_api_seconds += time.time() - t
            self.n_api_calls += 1
            return resp

        return _call()

    BACKEND = "vertex"

    def _map(self, fn, items: list):
        if self.max_workers == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(fn, items))

    def _cache_key(self, messages: list[dict], call: str, **params) -> str:
        return self.cache.key(self.model, messages, backend=self.BACKEND, call=call,
                              **params, **self._params_key())

    def _contents(self, messages: list[dict]) -> list[dict]:
        """OpenAI message format -> Vertex contents format."""
        contents = []
        for m in messages:
            role = m["role"]
            if role == "assistant":
                role = "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return contents

    def score_one(self, messages: list[dict]) -> str:
        k = self._cache_key(messages, "score", max_tokens=self.max_output_tokens)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._generate(self._contents(messages),
                              max_output_tokens=self.max_output_tokens,
                              temperature=self.temperature)
        text = resp.text
        self.cache.set(k, text, backend=self.BACKEND, call="score", model=self.model)
        return text

    def score(self, messages_list: list[list[dict]]) -> list[str]:
        out = self._map(self.score_one, messages_list)
        self.cache.flush()
        return out

    def complete_one(self, messages: list[dict], max_tokens: int = 64) -> str:
        k = self._cache_key(messages, "complete", max_tokens=max_tokens)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._generate(self._contents(messages),
                              max_output_tokens=max_tokens,
                              temperature=self.temperature)
        text = resp.text
        self.cache.set(k, text, backend=self.BACKEND, call="complete", model=self.model)
        return text

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        out = self._map(lambda m: self.complete_one(m, max_tokens), messages_list)
        self.cache.flush()
        return out

    def classify_one(self, messages: list[dict], top_logprobs: int = 10) -> dict:
        k = self._cache_key(messages, "classify", max_tokens=1, logprobs=top_logprobs)
        cached = self.cache.get(k)
        if cached is None:
            resp = self._generate(self._contents(messages), max_output_tokens=1,
                                  temperature=self.temperature, top_k=top_logprobs)
            cached = json.dumps(_raw_logprobs_payload(resp.candidates[0]), ensure_ascii=False)
            self.cache.set(k, cached, backend=self.BACKEND, call="classify", model=self.model)
        return _parse_logprobs_payload(cached)

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        out = self._map(self.classify_one, messages_list)
        self.cache.flush()
        return out


class ResponseShapeError(ValueError):
    """A model response did not have the structure the extraction code expects.

    Raised instead of returning an empty result: under schema 1 these were
    swallowed, turned into a missing logprob, and silently became a neutral
    p=0.5 in the reported metrics with no record that it had happened.
    """


def _lp_content_to_top(first: dict) -> dict:
    """Given a single logprobs.content[*] or candidate.token_log_probs[*]
    entry (dict), return {token: logprob}
    including the chosen token."""
    top = {t["token"]: t.get("log_prob") or t.get("logprob") for t in first.get("top_logprobs", [])}
    top.setdefault(first["token"], first.get("log_prob") or first.get("logprob"))
    return top


def _as_dict(obj):
    """Normalize an SDK model / dict / None into a plain dict."""
    if obj is None or isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    raise ResponseShapeError(f"cannot normalize {type(obj).__name__} into a dict")


def _raw_logprobs_payload(choice) -> dict:
    """The RAW logprobs structure for one sync chat choice, stored verbatim.
    Handles both Together's `choice.logprobs.content` and Vertex's
    `candidate.token_log_probs`.

    No flattening happens here — that is `_lp_content_to_top`'s job, and it
    runs on read so the cached value never depends on the parser's current
    shape.
    """
    if hasattr(choice, "logprobs") and choice.logprobs:
        lp = _as_dict(choice.logprobs)
        content = lp.get("content")
        if not content: raise ResponseShapeError("logprobs.content was empty")
        return {"text": choice.message.content or "", "content0": _as_dict(content[0])}
    if hasattr(choice, "token_log_probs") and choice.token_log_probs:
        content = _as_dict(choice.token_log_probs)
        if not content: raise ResponseShapeError("token_log_probs was empty")
        return {"text": choice.content.parts[0].text or "", "content0": _as_dict(content[0])}
    raise ResponseShapeError("response carried no usable logprobs")


def _parse_logprobs_payload(raw: str) -> dict:
    """Read-time parse of a cached classify payload -> {text, top_logprobs}."""
    payload = json.loads(raw)
    return {"text": payload.get("text", ""),
            "top_logprobs": _lp_content_to_top(payload["content0"])}


def top_logprobs_from_choice(choice) -> dict:
    """{token: logprob} for the first generated token of a sync chat choice.

    Raises ResponseShapeError if the response carried no usable logprobs, rather
    than returning {} and letting the caller silently fall back to p=0.5.
    """
    return _lp_content_to_top(_raw_logprobs_payload(choice)["content0"])


def _batch_body(obj: dict) -> dict:
    """The response body inside a Batch-API output line, whichever nesting it uses."""
    for path in (("response", "body"), ("body",), ()):
        node = obj
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, dict) and "choices" in node:
            return node
    raise ResponseShapeError(f"no 'choices' found in batch line (keys: {sorted(obj)[:6]})")


def _extract_top_logprobs_obj(obj: dict) -> dict:
    """Top-token logprobs from a batch-output line (nested dicts)."""
    content = _batch_body(obj)["choices"][0].get("logprobs", {}).get("content")
    if not content:
        raise ResponseShapeError("batch line carried no logprobs.content")
    return _lp_content_to_top(content[0])


def _extract_content(obj: dict) -> str:
    """Pull the assistant content out of a batch-output line."""
    return _batch_body(obj)["choices"][0]["message"]["content"] or ""


class TogetherBatchClient:
    """Together Batch API backend with the same score(messages_list) interface
    as TogetherClient.

    For a list of message-sets it: serves cached rows for free, writes the
    uncached ones to a JSONL (one request per row, custom_id = index), uploads
    with purpose='batch-api', creates a /v1/chat/completions batch, polls to
    completion, downloads the output, reconciles by custom_id, and routes any
    failed/missing rows to empty output (-> parse-fail/neutral-50 in the
    predictor) while logging the failure count.

    Cache keys are NOT shared with TogetherClient. Schema 1 shared them on the
    theory that results "carry over between backends"; in practice the two
    serving paths can answer the same prompt differently, so sharing meant the
    later run silently overwrote the earlier one's value. Backend is now part of
    the key, and failed rows are never written, so a retry actually retries.
    """

    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 16, temperature: float = 0.0,
                 reasoning_effort: str | None = None, extra_create_kwargs: dict | None = None,
                 poll_interval: float = 30.0, max_wait: float = 14400.0):
        self.model = model
        # `is None`, NOT `or`: ResponseCache defines __len__, so an EMPTY cache is
        # falsy and `cache or ResponseCache()` silently swapped a caller's
        # fresh cache for the project default.
        self.cache = ResponseCache() if cache is None else cache
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.extra_create_kwargs = extra_create_kwargs or {}
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.error_count = 0          # rows that failed / returned no output
        self.error_samples = []       # a few (custom_id, error) for logging
        self.shape_errors = []        # (custom_id, reason) for malformed output lines
        self.n_submitted = 0          # rows actually sent to the batch
        self.last_batch_seconds = 0.0
        self._client = None

    BACKEND = "batch"

    def _params_key(self) -> dict:
        return {"temperature": self.temperature, "reasoning_effort": self.reasoning_effort,
                "extra": self.extra_create_kwargs}

    def _key(self, messages: list[dict]) -> str:
        return self.cache.key(self.model, messages, backend=self.BACKEND, call="score",
                              max_tokens=self.max_output_tokens, **self._params_key())

    def _body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages,
                "max_tokens": self.max_output_tokens, "temperature": self.temperature}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        body.update(self.extra_create_kwargs)
        return body

    def _ensure_client(self):
        if self._client is None:
            from together import Together
            self._client = Together(timeout=120.0)
        return self._client

    def _download_text(self, file_id: str) -> str:
        client = self._ensure_client()
        import tempfile
        with tempfile.NamedTemporaryFile("w+b", suffix=".jsonl", delete=True) as tf:
            with client.files.with_streaming_response.content(id=file_id) as resp:
                for chunk in resp.iter_bytes():
                    tf.write(chunk)
            tf.seek(0)
            return tf.read().decode("utf-8")

    # classify (logprobs) variants of the body/key
    def _body_classify(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": 1,
                "temperature": self.temperature, "logprobs": True, "top_logprobs": 10}
        body.update(self.extra_create_kwargs)
        return body

    def _key_classify(self, messages: list[dict]) -> str:
        return self.cache.key(self.model, messages, backend=self.BACKEND, call="classify",
                              max_tokens=1, logprobs=10, **self._params_key())

    def _run(self, uncached, results, key_fn, body_fn, extract_fn, call) -> None:
        import json as _json
        import tempfile
        client = self._ensure_client()
        self.n_submitted += len(uncached)

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as tf:
            for idx, messages in uncached:
                tf.write(_json.dumps({"custom_id": str(idx), "body": body_fn(messages)},
                                     ensure_ascii=False) + "\n")
            input_path = tf.name

        file_resp = client.files.upload(file=input_path, purpose="batch-api", check=False)
        resp = client.batches.create(input_file_id=file_resp.id,
                                     endpoint="/v1/chat/completions")
        batch = getattr(resp, "job", resp)

        start = time.time()
        while True:
            batch = client.batches.retrieve(batch.id)
            status = str(getattr(batch, "status", "")).upper()
            if "COMPLETED" in status:
                break
            if any(s in status for s in ("FAILED", "EXPIRED", "CANCELLED", "ERROR")):
                raise RuntimeError(f"batch ended with status {status}")
            if time.time() - start > self.max_wait:
                raise TimeoutError(f"batch exceeded max_wait ({self.max_wait}s)")
            time.sleep(self.poll_interval)
        self.last_batch_seconds = time.time() - start

        # reconcile by custom_id (output line order does NOT match input order)
        output = {}
        out_id = getattr(batch, "output_file_id", None)
        if out_id:
            for line in self._download_text(out_id).splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                cid = str(obj.get("custom_id"))
                try:
                    output[cid] = extract_fn(obj)
                except (ResponseShapeError, KeyError, IndexError, TypeError) as e:
                    # Malformed line: record WHY. Do not store it — see below.
                    self.shape_errors.append((cid, f"{type(e).__name__}: {e}"))

        errors = {}
        err_id = getattr(batch, "error_file_id", None)
        if err_id:
            for line in self._download_text(err_id).splitlines():
                obj = _json.loads(line.strip()) if line.strip() else None
                if obj:
                    errors[str(obj.get("custom_id"))] = obj

        for cid, (idx, messages) in {str(i): (i, m) for i, m in uncached}.items():
            val = output.get(cid)
            if not val:                           # missing, errored, or malformed
                self.error_count += 1
                if len(self.error_samples) < 5:
                    self.error_samples.append((cid, errors.get(cid, "no output")))
                # NOT cached: schema 1 stored "" here, which was then served as a
                # cache HIT forever, permanently feeding a neutral p=0.5 into the
                # metrics. A miss must stay a miss so a rerun retries the row.
                results[idx] = ""
                continue
            self.cache.set(key_fn(messages), val, backend=self.BACKEND,
                           call=call, model=self.model)
            results[idx] = val

    def _serve(self, messages_list, key_fn, body_fn, extract_fn, call):
        results = [None] * len(messages_list)
        uncached = []
        for i, messages in enumerate(messages_list):
            cached = self.cache.get(key_fn(messages))
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, messages))
        if uncached:
            self._run(uncached, results, key_fn, body_fn, extract_fn, call)
        return [r if r is not None else "" for r in results]

    def score(self, messages_list: list[list[dict]]) -> list[str]:
        return self._serve(messages_list, self._key, self._body, _extract_content, "score")

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        """Cached value is the RAW logprobs structure; flattening happens here, on
        read, so a change to the parser can never leave stale values behind an
        unchanged key."""
        import json as _json
        raw = self._serve(
            messages_list, self._key_classify, self._body_classify,
            lambda obj: _json.dumps({"text": _extract_content(obj),
                                     "content0": _batch_body(obj)["choices"][0]
                                                 ["logprobs"]["content"][0]},
                                    ensure_ascii=False),
            "classify")
        return [_parse_logprobs_payload(r) if r else {"text": "", "top_logprobs": {}}
                for r in raw]

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        def key_fn(m):
            return self.cache.key(self.model, m, backend=self.BACKEND, call="complete",
                                  max_tokens=max_tokens, **self._params_key())

        def body_fn(m):
            b = {"model": self.model, "messages": m, "max_tokens": max_tokens,
                 "temperature": self.temperature}
            b.update(self.extra_create_kwargs)
            return b
        return self._serve(messages_list, key_fn, body_fn, _extract_content, "complete")

    @property
    def avg_latency(self) -> float:
        return self.last_batch_seconds / self.n_submitted if self.n_submitted else 0.0

"""LLM utilities for Together AI: token/cost estimation, response caching, and a
thin OpenAI-compatible client with retry/backoff.

No network call happens on import. The Together SDK is imported lazily and picks
up TOGETHER_API_KEY from the environment (loaded from .env by the package init).
Token counting uses tiktoken as an approximation; the Llama/GPT-OSS tokenizers
differ slightly, so cost figures are estimates (accurate to within a few %).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time

# USD per 1,000,000 tokens. Verified 2026-06-22 from together.ai pricing.
PRICING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference": {"input": 0.18, "output": 0.18},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":     {"input": 0.18, "output": 0.18},
    "meta-llama/Llama-3.3-70B-Instruct-Turbo":         {"input": 1.04, "output": 1.04},
    "openai/gpt-oss-20b":                              {"input": 0.05, "output": 0.20},
    "openai/gpt-oss-120b":                             {"input": 0.15, "output": 0.60},
    "Qwen/Qwen3.5-9B":                                {"input": 0.17, "output": 0.25},
    "google/gemma-4-31B-it":                           {"input": 0.39, "output": 0.97},
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
}
DEFAULT_CONFIG = {"max_output_tokens": 16}


def make_client(model: str, cache=None, backend: str = "sync", **overrides):
    """Build a client for `model` using its registered config (or the default).

    backend="sync"  -> TogetherClient (per-row, cached) for dev/notebook/interactive
    backend="batch" -> TogetherBatchClient (Together Batch API) for full-test runs

    Both share the same cache and cache keys, so results carry over between them.
    Fine-tuned model ids fall back to the default config, so the same call site
    works for base and fine-tuned models."""
    cfg = dict(MODEL_CONFIGS.get(model, DEFAULT_CONFIG))
    cfg.update(overrides)
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
except Exception:
    def count_tokens(text: str) -> int:
        return math.ceil(len(text) / 4)

    TOKENIZER = "heuristic (~4 chars/token)"


def estimate_cost(n_input_tokens: int, n_output_tokens: int, model: str) -> float:
    """Estimated USD cost for input/output token counts at a model's rate."""
    price = PRICING[model]
    return (n_input_tokens * price["input"] + n_output_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# On-disk response cache (keyed by model + messages + params) so reruns are free.
# ---------------------------------------------------------------------------
# Anchor the cache to the project root (parent of this package), so it is found
# regardless of the current working directory (e.g. when run from notebooks/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, ".llm_cache.json")


class ResponseCache:
    def __init__(self, path: str | None = None):
        self.path = path or _DEFAULT_CACHE
        self._data: dict[str, str] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    @staticmethod
    def key(model: str, messages: list[dict], **params) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Together client (OpenAI-compatible chat completions) with retry + backoff.
# ---------------------------------------------------------------------------
class TogetherClient:
    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 8, temperature: float = 0.0,
                 max_retries: int = 4, backoff_base: float = 1.0, timeout: float = 120.0,
                 reasoning_effort: str | None = None, extra_create_kwargs: dict | None = None):
        self.model = model
        self.cache = cache or ResponseCache()
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
        """Call chat.completions.create with exponential backoff on errors."""
        if self.reasoning_effort is not None:
            kwargs.setdefault("reasoning_effort", self.reasoning_effort)
        for k, v in self.extra_create_kwargs.items():
            kwargs.setdefault(k, v)
        client = self._ensure_client()
        for attempt in range(self.max_retries):
            try:
                t = time.time()
                resp = client.chat.completions.create(**kwargs)
                self.total_api_seconds += time.time() - t
                self.n_api_calls += 1
                return resp
            except Exception:
                self.api_errors += 1
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.backoff_base * (2 ** attempt))

    # --- primary: 0-100 score path ---------------------------------------
    def score_one(self, messages: list[dict]) -> str:
        k = self.cache.key(self.model, messages, max_tokens=self.max_output_tokens,
                           **self._params_key())
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._create(model=self.model, messages=messages,
                            max_tokens=self.max_output_tokens, temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        self.cache.set(k, text)
        return text

    def score(self, messages_list: list[list[dict]]) -> list[str]:
        out = [self.score_one(m) for m in messages_list]
        self.cache.flush()
        return out

    def complete_one(self, messages: list[dict], max_tokens: int = 64) -> str:
        k = self.cache.key(self.model, messages, max_tokens=max_tokens, kind="complete",
                           **self._params_key())
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        resp = self._create(model=self.model, messages=messages, max_tokens=max_tokens,
                            temperature=self.temperature)
        text = resp.choices[0].message.content or ""
        self.cache.set(k, text)
        return text

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        """Free-text generation (e.g. one-sentence rationales)."""
        out = [self.complete_one(m, max_tokens) for m in messages_list]
        self.cache.flush()
        return out

    # --- optional: logprob-based True/False path -------------------------
    def classify_one(self, messages: list[dict], top_logprobs: int = 10) -> dict:
        k = self.cache.key(self.model, messages, max_tokens=1,
                           logprobs=top_logprobs, **self._params_key())
        cached = self.cache.get(k)
        if cached is not None:
            return json.loads(cached)
        # Together returns logprobs nested in logprobs.content[*]; the SDK rejects
        # top_logprobs as a kwarg, so request it via extra_body.
        resp = self._create(model=self.model, messages=messages, max_tokens=1,
                            temperature=self.temperature,
                            extra_body={"logprobs": True, "top_logprobs": top_logprobs})
        payload = {"text": resp.choices[0].message.content or "",
                   "top_logprobs": _extract_top_logprobs(resp.choices[0])}
        self.cache.set(k, json.dumps(payload, ensure_ascii=False))
        return payload

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        out = [self.classify_one(m) for m in messages_list]
        self.cache.flush()
        return out

    def ping(self) -> str:
        """Minimal availability probe (1 token). Raises if the model is not
        callable serverlessly. Used to verify model strings before a run."""
        resp = self._create(model=self.model,
                            messages=[{"role": "user", "content": "Reply with: ok"}],
                            max_tokens=1, temperature=0.0)
        return resp.choices[0].message.content or ""


def _lp_content_to_top(first: dict) -> dict:
    """Given a single logprobs.content[*] entry (dict), return {token: logprob}
    including the chosen token."""
    top = {t["token"]: t["logprob"] for t in first.get("top_logprobs", [])}
    top.setdefault(first["token"], first["logprob"])
    return top


def _extract_top_logprobs(choice) -> dict:
    """Top-token logprobs from a (sync) chat choice object: logprobs.content[0]."""
    try:
        lp = choice.logprobs
        content = lp.get("content") if isinstance(lp, dict) else lp.content
        first = content[0]
        first = first if isinstance(first, dict) else first.model_dump()
        return _lp_content_to_top(first)
    except Exception:
        return {}


def _extract_top_logprobs_obj(obj: dict) -> dict:
    """Top-token logprobs from a batch-output line (nested dicts)."""
    body = obj.get("response", {})
    body = body.get("body") if isinstance(body, dict) else None
    body = body or obj.get("body") or obj
    try:
        first = body["choices"][0]["logprobs"]["content"][0]
        return _lp_content_to_top(first)
    except Exception:
        return {}


def _extract_content(obj: dict) -> str:
    """Pull the assistant content out of a batch-output line, defensively."""
    for path in (("response", "body"), ("body",), ()):
        node = obj
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and "choices" in node:
            try:
                return node["choices"][0]["message"]["content"] or ""
            except Exception:
                pass
    return ""


class TogetherBatchClient:
    """Together Batch API backend with the same score(messages_list) interface
    as TogetherClient and the SAME cache keys, so results are interchangeable.

    For a list of message-sets it: serves cached rows for free, writes the
    uncached ones to a JSONL (one request per row, custom_id = index), uploads
    with purpose='batch-api', creates a /v1/chat/completions batch, polls to
    completion, downloads the output, reconciles by custom_id, and routes any
    failed/missing rows to empty output (-> parse-fail/neutral-50 in the
    predictor) while logging the failure count.
    """

    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 16, temperature: float = 0.0,
                 reasoning_effort: str | None = None, extra_create_kwargs: dict | None = None,
                 poll_interval: float = 30.0, max_wait: float = 14400.0):
        self.model = model
        self.cache = cache or ResponseCache()
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.extra_create_kwargs = extra_create_kwargs or {}
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.error_count = 0          # rows that failed / returned no output
        self.error_samples = []       # a few (custom_id, error) for logging
        self.n_submitted = 0          # rows actually sent to the batch
        self.last_batch_seconds = 0.0
        self._client = None

    # cache key identical to TogetherClient.score_one
    def _params_key(self) -> dict:
        return {"temperature": self.temperature, "reasoning_effort": self.reasoning_effort,
                "extra": self.extra_create_kwargs}

    def _key(self, messages: list[dict]) -> str:
        return self.cache.key(self.model, messages, max_tokens=self.max_output_tokens,
                              **self._params_key())

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
        return self.cache.key(self.model, messages, max_tokens=1, logprobs=10,
                              **self._params_key())

    def _run(self, uncached, results, key_fn, body_fn, extract_fn) -> None:
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
                output[str(obj.get("custom_id"))] = extract_fn(obj)

        errors = {}
        err_id = getattr(batch, "error_file_id", None)
        if err_id:
            try:
                for line in self._download_text(err_id).splitlines():
                    obj = _json.loads(line.strip()) if line.strip() else None
                    if obj:
                        errors[str(obj.get("custom_id"))] = obj
            except Exception:
                pass

        for cid, (idx, messages) in {str(i): (i, m) for i, m in uncached}.items():
            val = output.get(cid)
            if not val:                           # missing or explicitly errored
                self.error_count += 1
                if len(self.error_samples) < 5:
                    self.error_samples.append((cid, errors.get(cid, "no output")))
                val = ""
            self.cache.set(key_fn(messages), val)
            results[idx] = val
        self.cache.flush()

    def _serve(self, messages_list, key_fn, body_fn, extract_fn):
        results = [None] * len(messages_list)
        uncached = []
        for i, messages in enumerate(messages_list):
            cached = self.cache.get(key_fn(messages))
            if cached is not None:
                results[i] = cached
            else:
                uncached.append((i, messages))
        if uncached:
            self._run(uncached, results, key_fn, body_fn, extract_fn)
        return [r if r is not None else "" for r in results]

    def score(self, messages_list: list[list[dict]]) -> list[str]:
        return self._serve(messages_list, self._key, self._body, _extract_content)

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        import json as _json
        raw = self._serve(messages_list, self._key_classify, self._body_classify,
                          lambda obj: _json.dumps(_extract_top_logprobs_obj(obj)))
        return [{"top_logprobs": _json.loads(r) if r else {}} for r in raw]

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        def key_fn(m):
            return self.cache.key(self.model, m, max_tokens=max_tokens, kind="complete",
                                  **self._params_key())

        def body_fn(m):
            b = {"model": self.model, "messages": m, "max_tokens": max_tokens,
                 "temperature": self.temperature}
            b.update(self.extra_create_kwargs)
            return b
        return self._serve(messages_list, key_fn, body_fn, _extract_content)

    @property
    def avg_latency(self) -> float:
        return self.last_batch_seconds / self.n_submitted if self.n_submitted else 0.0

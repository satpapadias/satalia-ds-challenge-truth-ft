"""LLM utilities for the Vertex AI (Gemini) backend: token/cost estimation,
response caching, and a thin client with retry/backoff.

No network call happens on import. The SDK is imported lazily and picks up its
credentials from the environment via gcloud ADC. Token counting uses tiktoken as
an approximation; tokenizers differ, so cost figures are estimates.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import time
import warnings

import numpy as np

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

# USD per 1,000,000 tokens. Vertex pricing verified 2026-08-14.
PRICING = {
    "gemini-2.5-flash":                                {"input": 0.35, "output": 0.70},
}

# Per-model client settings.
MODEL_CONFIGS = {
    "gemini-2.5-flash":     {"max_output_tokens": 16},
}
DEFAULT_CONFIG = {"max_output_tokens": 16}


def _retryable_errors():
    """Transient Vertex/HTTP failures worth retrying."""
    try:
        from google.api_core import exceptions as ge
        return (ge.ResourceExhausted, ge.ServiceUnavailable, ge.InternalServerError, ge.GatewayTimeout)
    except ImportError:                          # SDK absent: nothing to import
        return (ConnectionError, TimeoutError)


RETRYABLE_ERRORS = _retryable_errors()


def make_client(model: str, cache=None, **overrides):
    """Build a VertexClient for `model` using its registered config (or the default).

    Fine-tuned model ids fall back to the default config, so the same call site
    works for base and fine-tuned models."""
    cfg = dict(MODEL_CONFIGS.get(model, DEFAULT_CONFIG))
    cfg.update(overrides)
    return VertexClient(model, cache=cache, **cfg)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

    TOKENIZER = "tiktoken/o200k_base (approx for Llama/GPT-OSS)"
except Exception as _tiktoken_error:             # noqa: F841 - reported below
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
    price = PRICING.get(model)
    if price is None:
        if "gemini" in model:
            price = PRICING["gemini-2.5-flash"]
        else:
            price = {"input": 0.0, "output": 0.0}
    return (n_input_tokens * price["input"] + n_output_tokens * price["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# On-disk response cache so reruns are free.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, ".llm_cache")

CACHE_SCHEMA = 2


class ResponseCache:
    """Cache of raw model responses, keyed by everything that determines them."""

    def __init__(self, path: str | None = None):
        import diskcache
        self.path = path or _DEFAULT_CACHE
        self._cache = diskcache.Cache(self.path)

    @staticmethod
    def key(model: str, messages: list[dict], *, backend: str, call: str, **params) -> str:
        """Content hash of the full request identity."""
        blob = json.dumps({"cache_schema": CACHE_SCHEMA, "backend": backend, "call": call,
                           "model": model, "messages": messages, "params": params},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str):
        """Cached raw value, or None on a miss."""
        env = self._cache.get(key)
        return None if env is None else env["value"]

    def envelope(self, key: str):
        """Full stored record including provenance."""
        return self._cache.get(key)

    def set(self, key: str, value: str, *, backend: str, call: str, model: str) -> None:
        """Store a successful response with provenance."""
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
        """No-op: diskcache writes through on set()."""

    def close(self) -> None:
        self._cache.close()

    def __len__(self) -> int:
        return len(self._cache)


def _sdk_version() -> str:
    """Version string recorded in each cache envelope, for provenance."""
    versions = []
    try:
        import importlib.metadata as md
        return f"google-cloud-aiplatform {md.version('google-cloud-aiplatform')}"
    except md.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# Vertex client with retry + backoff.
# ---------------------------------------------------------------------------
class VertexClient:
    """LLM client for Vertex AI (Gemini)."""

    def __init__(self, model: str, cache: ResponseCache | None = None,
                 max_output_tokens: int = 8, temperature: float = 0.0,
                 max_retries: int = 4, backoff_base: float = 1.0, timeout: float = 120.0,
                 max_workers: int = 16, **kwargs):
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
                    "google-cloud-aiplatform not installed; run `uv pip install google-cloud-aiplatform`."
                ) from e
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION")
            if not project or not location:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set for Vertex AI.")
            vertexai.init(project=project, location=location)
            self._client = GenerativeModel(self.model)
        return self._client

    def _generate(self, contents: list[dict] | dict, generation_config: dict, system_instruction: str | None = None):
        from vertexai.generative_models import GenerativeModel
        self._ensure_client()
        model = GenerativeModel(self.model, system_instruction=system_instruction)

        def _count_error(retry_state):
            self.api_errors += 1

        @retry(retry=retry_if_exception_type(RETRYABLE_ERRORS),
               wait=wait_exponential_jitter(initial=self.backoff_base, max=60.0),
               stop=stop_after_attempt(self.max_retries),
               before_sleep=_count_error,
               reraise=True)
        def _call():
            t = time.time()
            resp = model.generate_content(
                contents,
                generation_config=generation_config,
            )
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

    def _contents(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """OpenAI message format -> Vertex contents format.
        
        Separates the system prompt for the new API format.
        """
        system_instruction = None
        contents = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_instruction = m["content"]
                continue
            if role == "assistant":
                role = "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system_instruction, contents

    def score_one(self, messages: list[dict]) -> str:
        k = self._cache_key(messages, "score", max_tokens=self.max_output_tokens)
        cached = self.cache.get(k)
        if cached is not None:
            return cached
        generation_config = {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens, "thinking_config": {"thinking_budget": 0}}
        system_instruction, contents = self._contents(messages)
        resp = self._generate(contents, generation_config, system_instruction)
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

        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": max_tokens,
            "thinking_config": {"thinking_budget": 0},
        }
        system_instruction, contents = self._contents(messages)
        resp = self._generate(contents, generation_config, system_instruction)
        text = resp.text
        self.cache.set(k, text, backend=self.BACKEND, call="complete", model=self.model)
        return text

    def complete(self, messages_list: list[list[dict]], max_tokens: int = 64) -> list[str]:
        out = self._map(lambda m: self.complete_one(m, max_tokens), messages_list)
        self.cache.flush()
        return out

    def classify_one(self, messages: list[dict]) -> dict:
        generation_config = {
            "temperature": 0,
            "max_output_tokens": 1,
            "top_k": 5,
            "response_logprobs": True,
            "logprobs": 5,
            "thinking_config": {"thinking_budget": 0}
        }

        cache_params = generation_config.copy()
        cache_params.pop("temperature", None)
        k = self._cache_key(messages, "classify", **cache_params)
        cached = self.cache.get(k)
        if cached:
            return json.loads(cached)

        system_instruction, contents = self._contents(messages)
        resp = self._generate(contents, generation_config, system_instruction)

        if not resp.candidates:
            raise ResponseShapeError("No candidates returned from model")
        candidate = resp.candidates[0]

        finish_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)

        if finish_name not in ("STOP", "MAX_TOKENS"):
            raise ResponseShapeError(f"Generation failed with reason: {finish_name}")

        if not hasattr(candidate, "logprobs_result") or not candidate.logprobs_result:
             raise ResponseShapeError("Response did not contain logprobs_result attribute.")
        if not hasattr(candidate.logprobs_result, "top_candidates") or not candidate.logprobs_result.top_candidates:
             raise ResponseShapeError("Response did not contain top_candidates.")

        first_token_logprobs = candidate.logprobs_result.top_candidates[0]
        
        true_lps = []
        false_lps = []
        
        for cand in first_token_logprobs.candidates:
            tok_str = str(getattr(cand, 'token', '')).strip().strip('\'"._').lower()
            tok_id = getattr(cand, 'token_id', None)
            
            if tok_str == 'true' or tok_id == 4339:
                true_lps.append(cand.log_probability)
            elif tok_str == 'false' or tok_id == 9277:
                false_lps.append(cand.log_probability)

        def safe_logsumexp(lps: list[float]) -> float:
            if not lps:
                return -100.0
            arr = np.array(lps)
            max_lp = np.max(arr)
            return float(max_lp + np.log(np.sum(np.exp(arr - max_lp))))

        result = {
            "text": "True",
            "top_logprobs": {
                "True": safe_logsumexp(true_lps), 
                "False": safe_logsumexp(false_lps)
            }
        }
        self.cache.set(k, json.dumps(result), backend=self.BACKEND, call="classify", model=self.model)
        return result

    def classify(self, messages_list: list[list[dict]]) -> list[dict]:
        out = self._map(self.classify_one, messages_list)
        self.cache.flush()
        return out


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
    """A model response did not have the structure the extraction code expects."""

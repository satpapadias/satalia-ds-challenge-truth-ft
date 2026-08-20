"""Shared plumbing: request identity, structured logging, auth, and card serving.

Everything here is used by all four agents, so it is the one place to change how
a request is identified, how a log entry is shaped, or how a token is checked.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import secrets
import sys
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Request identity
# ---------------------------------------------------------------------------
# Two identifiers travel with every request, and they answer different questions.
#
#   traceparent  W3C Trace Context. The transport-level trace, understood by
#                hosted log and trace backends without a custom correlator, so
#                one request can be followed across process boundaries.
#   run_id       Our own logical identifier for a single verification run. It
#                survives even when tracing is switched off, and it is what the
#                A2A context_id is set to so the agents' own task records line
#                up with the logs.
_TRACEPARENT = contextvars.ContextVar("traceparent", default="")
_RUN_ID = contextvars.ContextVar("run_id", default="")
_AGENT_NAME = contextvars.ContextVar("agent_name", default="")

# version-traceid-spanid-flags, per the W3C Trace Context recommendation.
_TRACEPARENT_RE = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def new_traceparent() -> str:
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def child_traceparent(parent: str) -> str:
    """A new span under the same trace.

    The trace id is preserved and the span id replaced, so every hop is a
    distinct span that a backend can still assemble into one trace. Reusing the
    parent's span id wholesale would flatten the fan-out into a single span and
    lose exactly the structure that makes it debuggable.
    """
    m = _TRACEPARENT_RE.match(parent or "")
    if not m:
        return new_traceparent()
    return f"00-{m.group(1)}-{secrets.token_hex(8)}-{m.group(3)}"


def trace_id(traceparent: str = "") -> str:
    m = _TRACEPARENT_RE.match(traceparent or get_traceparent())
    return m.group(1) if m else ""


def get_traceparent() -> str:
    return _TRACEPARENT.get()


def get_run_id() -> str:
    return _RUN_ID.get()


def bind_request(traceparent: str = "", run_id: str = "") -> None:
    """Attach identity to the current context. Called once per inbound request."""
    _TRACEPARENT.set(traceparent or new_traceparent())
    _RUN_ID.set(run_id or secrets.token_hex(8))


def bind_agent(name: str) -> None:
    _AGENT_NAME.set(name)


def outbound_headers() -> dict[str, str]:
    """Headers that carry request identity to the next hop.

    Used for both the A2A hop and the MCP hop. The MCP hop is the one that would
    otherwise be invisible, and it is where the provider spend happens, so a
    trace that stops at the agent boundary cannot answer which tool call was
    slow or expensive.
    """
    return {"traceparent": child_traceparent(get_traceparent()),
            "x-truthclf-run-id": get_run_id()}


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """One JSON object per line on stdout.

    Hosted log collectors ingest this without a sidecar, and the trace fields
    use the names a Google Cloud Logging backend recognises so entries are
    correlated to their trace automatically rather than by text search.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "agent": _AGENT_NAME.get(),
            "run_id": _RUN_ID.get(),
            "logging.googleapis.com/spanId": _span_id(),
        }
        tid = trace_id()
        if tid:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            entry["logging.googleapis.com/trace"] = (
                f"projects/{project}/traces/{tid}" if project else tid)
        extra = getattr(record, "fields", None)
        if extra:
            entry.update(extra)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _span_id() -> str:
    m = _TRACEPARENT_RE.match(get_traceparent())
    return m.group(2) if m else ""


def configure_logging(agent_name: str, level: str = "INFO") -> logging.Logger:
    bind_agent(agent_name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn's own access log is plain text and would break the JSON stream.
    logging.getLogger("uvicorn.access").disabled = True
    return logging.getLogger(agent_name)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    logger.info(message, extra={"fields": fields})


class timed:
    """Context manager measuring a block, for the duration field on a log entry."""

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = round((time.perf_counter() - self._t) * 1000, 1)
        return False


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class BearerAuth:
    """Shared-secret bearer check.

    Constant-time comparison, because a token check that returns early on the
    first differing byte leaks the token's prefix to anyone who can time it.

    A missing token in the environment is refused at start-up rather than
    treated as "no auth required": an agent that silently serves unauthenticated
    when misconfigured is worse than one that will not start.
    """

    def __init__(self, env_var: str, *, required: bool = True):
        self.env_var = env_var
        self.token = os.environ.get(env_var, "")
        if required and not self.token:
            raise RuntimeError(
                f"{env_var} is not set. Refusing to start rather than serving "
                "an unauthenticated endpoint.")

    def check(self, request: Request) -> bool:
        # 1. Try checking the standard Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header:
            scheme, _, value = auth_header.partition(" ")
            if scheme.lower() == "bearer":
                if secrets.compare_digest(value.strip(), self.token):
                    return True

        # 2. Fallback for proxies that strip/clash Authorization, like Cloud Run's IAM.
        api_key_header = request.headers.get("x-api-key", "")
        if api_key_header:
            if secrets.compare_digest(api_key_header.strip(), self.token):
                return True

        return False


def unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": "unauthorized",
         "detail": "supply a bearer token: 'Authorization: Bearer <token>' or 'X-API-Key: <token>'"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"})


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------
def resolve_base_url(request: Request) -> str:
    """The externally reachable base URL for this agent.

    An agent card is a contract: it tells another agent where to call. A card
    advertising an address the agent does not answer on fails at the caller with
    nothing logged on the serving side, so the address is derived from the
    request that is demonstrably reaching us rather than from configuration that
    may be stale.

    Managed container platforms assign the URL at deploy time, so it cannot be
    baked into an image. PUBLIC_BASE_URL overrides when the externally visible
    address differs from what the proxy reports.
    """
    override = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    forwarded_host = request.headers.get("x-forwarded-host")
    host = forwarded_host or request.headers.get("host") or request.url.netloc
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return f"{proto}://{host}".rstrip("/")


def env_url(name: str, default: str = "") -> str:
    return os.environ.get(name, default).rstrip("/")
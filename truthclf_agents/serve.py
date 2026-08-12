"""Building an A2A agent application, and moving JSON through A2A messages.

Every agent is one FastAPI app exposing:

    GET  /.well-known/agent-card.json   capability discovery
    POST /                              A2A JSON-RPC (message/send, message/stream,
                                        tasks/get, tasks/cancel)
    GET  /healthz                       container probe

The card route and the health probe are plain HTTP by necessity -- the first is
A2A's own discovery mechanism, the second is infrastructure. Everything an agent
says to another agent goes through the JSON-RPC route.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any, Callable

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, DEFAULT_RPC_URL
from fastapi import FastAPI
from google.protobuf import json_format, struct_pb2
from starlette.requests import Request
from starlette.responses import JSONResponse

from .cards import card_route
from .common import (BearerAuth, bind_request, get_run_id, log_event,
                     new_traceparent, unauthorized)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON in and out of A2A message parts
# ---------------------------------------------------------------------------
# A2A parts carry either text or a protobuf Value. Structured payloads use the
# Value, so a batch of statements survives the hop as data rather than as prose
# that something downstream has to re-parse.
def data_part(payload: dict) -> Part:
    value = struct_pb2.Value()
    json_format.ParseDict(payload, value)
    return Part(data=value)


def text_part(text: str) -> Part:
    return Part(text=text)


def read_payload(message: Message) -> dict:
    """The structured payload of an inbound message.

    Data parts are preferred. A text part is accepted as a fallback only if it
    parses as a JSON object, which keeps a hand-written curl usable for
    debugging without making prose a supported input.
    """
    for part in message.parts:
        if part.HasField("data"):
            return json_format.MessageToDict(part.data)
    for part in message.parts:
        if part.text:
            try:
                parsed = json.loads(part.text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(
        "message carried no JSON payload: expected a data part, or a text part "
        "containing a JSON object")


def agent_message(payload: dict, *, context_id: str, task_id: str,
                  summary: str = "") -> Message:
    parts = [data_part(payload)]
    if summary:
        parts.append(text_part(summary))
    return Message(role=Role.ROLE_AGENT, parts=parts,
                   context_id=context_id, task_id=task_id)


# ---------------------------------------------------------------------------
# Executor base
# ---------------------------------------------------------------------------
class JsonAgentExecutor(AgentExecutor):
    """Runs a JSON-in, JSON-out handler through the A2A task lifecycle.

    Subclasses implement `handle`. This class owns the state transitions --
    submitted, working, then completed or failed -- so every agent reports the
    same lifecycle and a failure is never left as a task stuck in `working`.
    """

    agent_name = "agent"

    async def handle(self, payload: dict, context: RequestContext) -> tuple[dict, str]:
        """Return (result payload, one-line human summary)."""
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        # The A2A context id is adopted as the run identifier, so an agent's own
        # task records and its log entries share one key with the orchestrator's.
        bind_request(traceparent=new_traceparent(), run_id=context.context_id)

        # The first event for a new task must be the Task itself. A status
        # update sent before it is rejected, because there is no task yet for
        # the update to apply to -- the executor, not the updater, is what
        # brings a task into existence.
        if context.current_task is None:
            await event_queue.enqueue_event(Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message]))
        await updater.start_work()
        try:
            payload = read_payload(context.message)
        except ValueError as e:
            await updater.failed(message=agent_message(
                {"error": "InvalidPayload", "detail": str(e)},
                context_id=context.context_id, task_id=context.task_id))
            return
        try:
            result, summary = await self.handle(payload, context)
        except Exception as e:
            # The class name is the contract: a caller can act on
            # FineTunedRowNotCached but not on a generic execution failure.
            log_event(logger, "task failed", agent=self.agent_name,
                      error=type(e).__name__, detail=str(e)[:500])
            await updater.failed(message=agent_message(
                {"error": type(e).__name__, "detail": str(e)},
                context_id=context.context_id, task_id=context.task_id))
            return
        await updater.complete(message=agent_message(
            result, context_id=context.context_id, task_id=context.task_id,
            summary=summary))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
def build_app(*, agent_name: str, executor: AgentExecutor, card_builder: Callable,
              auth: BearerAuth, extra_routes: list | None = None,
              on_startup: Callable | None = None) -> FastAPI:
    """One agent, one app.

    The card is served from our own route rather than the SDK's helper: the
    SDK's card_modifier hook receives only the card and never the request, so it
    cannot derive the advertised address from the host that was actually
    reached. The card is still serialised by the SDK, so the wire format is the
    protocol's.
    """
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Reachability of the tool server is checked once at start-up, so a
        # misconfigured tool URL shows up in the boot log rather than as a
        # failure on whichever request arrives first.
        if on_startup:
            await on_startup()
        yield

    app = FastAPI(title=agent_name, docs_url=None, redoc_url=None,
                  lifespan=lifespan)

    # A card must be discoverable before a caller has been issued a token --
    # that is the point of discovery -- so it and the health probe are exempt.
    # Everything else requires the bearer token.
    exempt = {AGENT_CARD_WELL_KNOWN_PATH, "/healthz"}

    @app.middleware("http")
    async def auth_and_trace(request: Request, call_next):
        # Continue an inbound trace when the caller supplied one, so a fan-out
        # is a single trace rather than one per hop.
        bind_request(traceparent=request.headers.get("traceparent", ""),
                     run_id=request.headers.get("x-truthclf-run-id", ""))
        if request.url.path not in exempt and not auth.check(request):
            log_event(logger, "unauthorized", path=request.url.path)
            return unauthorized()
        response = await call_next(request)
        response.headers["x-truthclf-run-id"] = get_run_id()
        return response

    handler = DefaultRequestHandler(
        agent_executor=executor,
        # Per-instance task state. Adequate while an agent runs as a single
        # instance and every task completes inside one request; a shared store
        # is required before scaling out, or a tasks/get can land on an instance
        # that never saw the task.
        task_store=InMemoryTaskStore(),
        agent_card=card_builder(""),
    )

    app.add_api_route(AGENT_CARD_WELL_KNOWN_PATH, card_route(card_builder),
                      methods=["GET"], include_in_schema=False)
    app.add_api_route("/healthz", _health, methods=["GET"], include_in_schema=False)
    for route in (extra_routes or []):
        app.router.routes.append(route)

    add_a2a_routes_to_fastapi(
        app, jsonrpc_routes=create_jsonrpc_routes(handler, DEFAULT_RPC_URL))

    return app


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def run(app: FastAPI, host: str, port: int, agent_name: str) -> None:
    print(f"{agent_name} listening on http://{host}:{port}"
          f"  card: http://{host}:{port}{AGENT_CARD_WELL_KNOWN_PATH}", flush=True)
    uvicorn.run(app, host=host, port=port, log_config=None)

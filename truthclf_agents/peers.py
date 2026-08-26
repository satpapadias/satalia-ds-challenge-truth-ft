"""Talking to another agent over A2A.

Capability discovery happens once at start-up: each peer's card is fetched and
its client built from it. Fetching a card per request would add a round trip for
information that changes only when a peer is redeployed.
"""

from __future__ import annotations
import httpx

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx2
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import (Message, Part, Role, SendMessageRequest, TaskState)
from google.protobuf import json_format, struct_pb2

from .common import log_event, outbound_headers, timed

logger = logging.getLogger(__name__)

# States after which no further events will arrive for a task.
TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}


class PeerError(RuntimeError):
    """A peer agent could not be reached, timed out, or failed its task."""

    def __init__(self, peer: str, status: str, detail: str):
        self.peer = peer
        self.status = status          # "timeout" | "error"
        self.detail = detail
        super().__init__(f"{peer}: {status}: {detail}")


@dataclass
class PeerReply:
    """What a peer returned, with the task state it ended in."""

    payload: dict
    state: str
    task_id: str
    context_id: str
    summary: str = ""


@dataclass
class Peer:
    """One remote agent, discovered and ready to call."""

    name: str
    base_url: str
    client: Any
    card: Any
    streaming: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ask(self, payload: dict, *, context_id: str,
                  timeout: float = 300.0) -> PeerReply:
        """Send one message and collect the peer's final result."""
        value = struct_pb2.Value()
        json_format.ParseDict(payload, value)
        message = Message(
            message_id=uuid.uuid4().hex,
            role=Role.ROLE_USER,
            parts=[Part(data=value)],
            context_id=context_id)
        request = SendMessageRequest(message=message)

        try:
            with timed() as t:
                reply = await asyncio.wait_for(
                    self._collect(request), timeout=timeout)
        except TimeoutError as e:
            log_event(logger, "peer timeout", peer=self.name, timeout_s=timeout)
            raise PeerError(self.name, "timeout",
                            f"no result within {timeout:.0f}s") from e
        except Exception as e:
            log_event(logger, "peer error", peer=self.name,
                      error=type(e).__name__, detail=str(e)[:300])
            raise PeerError(self.name, "error", f"{type(e).__name__}: {e}") from e

        log_event(logger, "peer replied", peer=self.name, state=reply.state,
                  duration_ms=t.ms, task_id=reply.task_id)
        if reply.state != TaskState.Name(TaskState.TASK_STATE_COMPLETED):
            detail = reply.payload.get("detail") or reply.payload.get("error") or reply.state
            raise PeerError(self.name, "error", f"task ended {reply.state}: {detail}")
        return reply

    async def _collect(self, request: SendMessageRequest) -> PeerReply:
        """Reduce the event stream to the final result."""
        payload: dict = {}
        summary = ""
        state = TaskState.Name(TaskState.TASK_STATE_UNSPECIFIED)
        task_id = context_id = ""

        async for event in self.client.send_message(request):
            which = event.WhichOneof("payload")
            if which == "task":
                task = event.task
                task_id, context_id = task.id, task.context_id
                state = TaskState.Name(task.status.state)
                if task.status.state in TERMINAL:
                    payload, summary = _read_parts(task.status.message)
            elif which == "status_update":
                update = event.status_update
                task_id = task_id or update.task_id
                context_id = context_id or update.context_id
                state = TaskState.Name(update.status.state)
                if update.status.state in TERMINAL:
                    payload, summary = _read_parts(update.status.message)
            elif which == "message":
                payload, summary = _read_parts(event.message)

        return PeerReply(payload=payload, state=state, task_id=task_id,
                         context_id=context_id, summary=summary)


def _read_parts(message) -> tuple[dict, str]:
    data: dict = {}
    text = ""
    for part in getattr(message, "parts", []):
        if part.HasField("data"):
            data = json_format.MessageToDict(part.data)
        elif part.text:
            text = part.text
    return data, text


async def discover(name: str, base_url: str, timeout: float = 300.0) -> Peer:
    """Fetch a peer's card and build a client from it."""
    base_url = base_url.rstrip("/")

    # Fetch token explicitly, completely bypassing httpx.Auth bugs
    is_local = "127.0.0.1" in base_url or "localhost" in base_url
    gcp_token = None
    
    if not is_local:
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            auth_req = google.auth.transport.requests.Request()
            gcp_token = google.oauth2.id_token.fetch_id_token(auth_req, base_url)
        except Exception as e:
            logger.error(f"Explicit token fetch failed for {base_url}: {e}")

    async def inject_auth_and_trace(request: httpx.Request):
        # Attach the token natively to the outgoing headers
        if gcp_token:
            request.headers["Authorization"] = f"Bearer {gcp_token}"
            
        for key, value in outbound_headers().items():
            request.headers[key] = value

    http = httpx.AsyncClient(
        timeout=timeout,
        event_hooks={"request": [inject_auth_and_trace]},
    )

    card = await A2ACardResolver(http, base_url).get_agent_card()
    client = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(card)
    log_event(logger, "peer discovered", peer=card.name, url=base_url,
              streaming=card.capabilities.streaming,
              skills=[s.id for s in card.skills])
    return Peer(name=card.name, base_url=base_url, client=client, card=card,
                streaming=card.capabilities.streaming)
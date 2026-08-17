"""MCP client used by the agents to reach their tools.

This is the agent-to-tool half of the system. Agents never import truthclf; every
capability is a tool call to one of the MCP servers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .common import log_event, outbound_headers, timed
from .gcp_auth import GcpAuth

logger = logging.getLogger(__name__)


class ToolCallFailed(RuntimeError):
    """A tool reported an error. The server's message is preserved verbatim.

    Tool errors carry the originating class name, so the text is the most
    precise description of the failure available and is passed through rather
    than replaced with a generic one.
    """

    def __init__(self, tool: str, message: str):
        self.tool = tool
        self.message = message
        super().__init__(f"{tool}: {message}")


async def call_tool(server_url: str, tool: str, arguments: dict,
                    *, timeout: float = 300.0) -> Any:
    """Call one MCP tool and return its parsed result.

    A session is opened per call. The alternative -- one long-lived session per
    agent -- would save a handshake, but a dropped session then fails every
    subsequent request until the agent is restarted, and these calls are already
    dominated by model latency rather than connection setup.

    Request identity is injected as default headers on the HTTP client, so the
    tool call appears under the same trace as the A2A hop that caused it.
    """
    headers = outbound_headers()
    auth = GcpAuth()
    try:
        async with httpx2.AsyncClient(
            headers=headers, auth=auth, timeout=timeout
        ) as http_client:
            async with streamable_http_client(server_url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    with timed() as t:
                        result = await session.call_tool(tool, arguments)
    except BaseExceptionGroup as group:
        # The transport runs inside a task group, so anything raised in these
        # blocks arrives wrapped. Left wrapped, a caller's `except
        # ToolCallFailed` never matches and a recoverable condition is reported
        # as an unhandled error with no usable detail.
        raise _flatten(group) from group

    # Deliberately outside the blocks above: raising inside them would put this
    # error through the same wrapping.
    payload = _unwrap(tool, result)
    log_event(logger, "mcp tool call", tool=tool, server=server_url,
              duration_ms=t.ms, ok=True)
    return payload


def _flatten(group: BaseException) -> BaseException:
    """The most specific exception inside a (possibly nested) group."""
    while isinstance(group, BaseExceptionGroup) and group.exceptions:
        group = group.exceptions[0]
    return group


def _unwrap(tool: str, result) -> Any:
    """Turn a tool result into data, or raise with the server's own message."""
    if result.is_error:
        text = result.content[0].text if result.content else "no detail"
        raise ToolCallFailed(tool, text)
    if result.structured_content is not None:
        return result.structured_content
    if not result.content:
        raise ToolCallFailed(tool, "empty result with no error flag")
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ToolCallFailed(tool, f"result was not JSON: {text[:200]}") from e


async def probe(server_url: str) -> list[str]:
    """Tool names exposed by a server. Used as a start-up reachability check."""
    async with httpx2.AsyncClient(
        auth=GcpAuth(), timeout=30.0
    ) as http_client:
        async with streamable_http_client(server_url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [t.name for t in (await session.list_tools()).tools]

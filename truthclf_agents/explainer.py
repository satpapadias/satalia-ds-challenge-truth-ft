"""Explainer agent.

Wraps the explainer, reached as an MCP tool call to model-tools. It explains the
zero-shot predictor by default, which is the only predictor whose counterfactual
queries can currently be answered: occlusion needs the model re-scored with each
metadata field removed, and recorded probabilities cannot answer that.

This agent does not vote. It runs after the verdict is decided and contributes a
driver and a rationale to the response, never to the decision.

Run:  python -m truthclf_agents.explainer [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from a2a.server.agent_execution import RequestContext

from . import mcp_client
from .cards import explainer_card
from .common import BearerAuth, configure_logging, env_url, log_event, timed
from .serve import JsonAgentExecutor, build_app, run

AGENT_NAME = "truthclf-explainer"
MODEL_TOOLS_URL = env_url("MODEL_TOOLS_URL", "http://127.0.0.1:8082/mcp")

logger = logging.getLogger(AGENT_NAME)

class ExplainerExecutor(JsonAgentExecutor):
    agent_name = AGENT_NAME

    async def handle(self, payload: dict, context: RequestContext) -> tuple[dict, str]:
        points = payload.get("points") or []
        arguments = {
            # Defaulted to zero_shot, but fine_tuned is now a valid option as
            # it can be served live. A caller can explicitly request
            # "model": "fine_tuned" to explain it.
            "model": payload.get("model", "zero_shot"),
            "points": points,
            "with_rationale": payload.get("with_rationale", True),
            "driver_eps": payload.get("driver_eps", 0.05),
            "elicitation": payload.get("elicitation", "score"),
            # Left at the tool's strict default: any neutral fallback fails.
            # A tolerance measured on one fixed 1,800-row sample is not the same
            # quantity when applied per request at an arbitrary batch size -- on
            # a seven-row batch it would permit nothing anyway, and on a large
            # one it would silently admit failures this agent never measured.
            # A caller who knows its batch contains acceptable refusals can pass
            # a rate explicitly.
            "max_parse_failure_rate": payload.get("max_parse_failure_rate", 0.1),
        }
        if payload.get("labels") is not None:
            arguments["labels"] = payload["labels"]
            arguments["scheme"] = payload.get("scheme", "primary")

        with timed() as t:
            result = await mcp_client.call_tool(MODEL_TOOLS_URL, "explain", arguments)

        log_event(logger, "explained", n=result["n"], duration_ms=t.ms,
                  provider_calls=result["n_predictions"],
                  parse_failures=result["parse_failures"])

        drivers = result["aggregate"]["driver_distribution"]
        top = max(drivers, key=drivers.get) if drivers else "none"
        return result, (f"Explained {result['n']} statement(s) by field occlusion; "
                        f"most common driver: {top}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=AGENT_NAME)
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9103)))
    args = ap.parse_args()

    configure_logging(AGENT_NAME)

    async def startup():
        async def probe_loop():
            delay = 1.0
            while True:
                try:
                    tools = await mcp_client.probe(MODEL_TOOLS_URL)
                    log_event(logger, "connected to model-tools", url=MODEL_TOOLS_URL, tools=tools)
                    break
                except Exception as e:
                    log_event(logger, "model-tools probe failed, retrying...", url=MODEL_TOOLS_URL,
                              error=type(e).__name__, detail=str(e)[:150])
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

        asyncio.create_task(probe_loop())

    app = build_app(agent_name=AGENT_NAME, executor=ExplainerExecutor(),
                    card_builder=explainer_card, auth=None, on_startup=startup)
    run(app, args.host, args.port, AGENT_NAME)


if __name__ == "__main__":
    main()

"""Zero-shot predictor agent.

Wraps the zero-shot predictor. It holds no model logic of its own: prediction is
an MCP tool call to the model-tools server, which is the only process that
imports the truthclf package.

Run:  python -m truthclf_agents.zero_shot [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

from a2a.server.agent_execution import RequestContext

from . import mcp_client
from .cards import zero_shot_card
from .common import BearerAuth, configure_logging, env_url, log_event, timed
from .serve import JsonAgentExecutor, build_app, run

AGENT_NAME = "truthclf-zero-shot-predictor"
MODEL_TOOLS_URL = env_url("MODEL_TOOLS_URL", "http://127.0.0.1:8082/mcp")

logger = logging.getLogger(AGENT_NAME)


class ZeroShotExecutor(JsonAgentExecutor):
    agent_name = AGENT_NAME

    async def handle(self, payload: dict, context: RequestContext) -> tuple[dict, str]:
        points = payload.get("points") or []
        arguments = {
            "model": "zero_shot",
            "points": points,
            "elicitation": payload.get("elicitation", "logprob"),
            "calibrated": payload.get("calibrated", True),
        }
        # Labels are forwarded only when supplied, so the tool computes metrics
        # for its own predictions exactly as the package's predict() does.
        if payload.get("labels") is not None:
            arguments["labels"] = payload["labels"]
            arguments["scheme"] = payload.get("scheme", "primary")

        with timed() as t:
            result = await mcp_client.call_tool(MODEL_TOOLS_URL, "predict", arguments)

        log_event(logger, "predicted", n=result["n"], duration_ms=t.ms,
                  cache_hits=result["provenance"]["cache_hits"],
                  live_calls=result["provenance"]["live_calls"],
                  parse_failures=result["parse_failures"])

        return result, (f"Scored {result['n']} statement(s) with the zero-shot "
                        f"predictor at threshold {result['threshold']:.4f}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=AGENT_NAME)
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9101)))
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

    app = build_app(agent_name=AGENT_NAME, executor=ZeroShotExecutor(),
                    card_builder=zero_shot_card, auth=None, on_startup=startup)
    run(app, args.host, args.port, AGENT_NAME)


if __name__ == "__main__":
    main()

"""Fine-tuned predictor agent.

Wraps the fine-tuned predictor, reached as an MCP tool call to model-tools.

Coverage is partial and that is a normal condition, not a fault. The fine-tuned
model has no live endpoint, so it answers from probabilities recorded during its
evaluation, covering the validation and test splits. A statement outside that
set cannot be scored, and this agent reports that as an explicit per-statement
status rather than failing the batch or substituting another predictor.

Run:  python -m truthclf_agents.fine_tuned [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import logging
import os

from a2a.server.agent_execution import RequestContext

from . import mcp_client
from .cards import fine_tuned_card
from .common import BearerAuth, configure_logging, env_url, log_event, timed
from .mcp_client import ToolCallFailed
from .serve import JsonAgentExecutor, build_app, run

AGENT_NAME = "truthclf-fine-tuned-predictor"
MODEL_TOOLS_URL = env_url("MODEL_TOOLS_URL", "http://127.0.0.1:8082/mcp")

logger = logging.getLogger(AGENT_NAME)

# Tool errors that mean "this agent cannot score these statements", as distinct
# from "something went wrong". Both are reported to the caller as an unavailable
# result with a reason, because from the orchestrator's point of view they have
# the same consequence: no probability to reconcile with.
_UNAVAILABLE = ("FineTunedRowNotCached", "FineTunedModelNotServable")


class FineTunedExecutor(JsonAgentExecutor):
    agent_name = AGENT_NAME

    async def handle(self, payload: dict, context: RequestContext) -> tuple[dict, str]:
        points = payload.get("points") or []
        arguments = {
            "model": "fine_tuned",
            "points": points,
            "fine_tuned_source": payload.get("source", "cached"),
            "elicitation": "logprob",
            # Score the statements that have a recorded probability and name the
            # rest, rather than refusing a batch because one statement is new.
            # Partial coverage is this predictor's normal condition.
            "on_missing": "omit",
            "calibrated": payload.get("calibrated", True),
        }
        if payload.get("labels") is not None:
            arguments["labels"] = payload["labels"]
            arguments["scheme"] = payload.get("scheme", "primary")

        try:
            with timed() as t:
                result = await mcp_client.call_tool(
                    MODEL_TOOLS_URL, "predict", arguments)
        except ToolCallFailed as e:
            reason = _classify(e.message)
            if reason is None:
                raise
            # A whole-batch unavailability, reported as a result rather than a
            # task failure. The orchestrator needs to know these statements have
            # no fine-tuned probability, which is information, not an error.
            log_event(logger, "batch unavailable", reason=reason, n=len(points))
            return ({"status": "unavailable", "reason": reason,
                     "detail": e.message, "n": len(points), "predictions": []},
                    f"No fine-tuned probability available for these "
                    f"{len(points)} statement(s): {reason}.")

        log_event(logger, "predicted", n=result["n"], duration_ms=t.ms,
                  served_by=result["provenance"]["served_by"])

        result["status"] = "ok"
        return result, (f"Scored {result['n']} statement(s) with the fine-tuned "
                        f"predictor ({result['provenance']['served_by']}).")


def _classify(message: str) -> str | None:
    for name in _UNAVAILABLE:
        if name in message:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=AGENT_NAME)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9102)))
    args = ap.parse_args()

    configure_logging(AGENT_NAME)
    auth = BearerAuth("AGENT_TOKEN")

    async def startup():
        tools = await mcp_client.probe(MODEL_TOOLS_URL)
        log_event(logger, "connected to model-tools", url=MODEL_TOOLS_URL, tools=tools)

    app = build_app(agent_name=AGENT_NAME, executor=FineTunedExecutor(),
                    card_builder=fine_tuned_card, auth=auth, on_startup=startup)
    run(app, args.host, args.port, AGENT_NAME)


if __name__ == "__main__":
    main()

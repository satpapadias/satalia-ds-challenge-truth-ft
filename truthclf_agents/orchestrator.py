"""Orchestrator agent: the entry point.

Receives a set of statements and optional labels, delegates prediction to the
two predictor agents over A2A, reconciles their answers, asks the explainer for
per-statement explanations, and returns one verdict per statement with aggregate
metrics when labels were supplied.

It reaches prediction only through the predictor agents. It is not configured
with the model-tools address at all, so the shortcut of calling the prediction
tool directly is unavailable rather than merely discouraged. The one tool it
does use is metric computation, which is a tool and not an agent capability.

Run:  python -m truthclf_agents.orchestrator [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid

from a2a.server.agent_execution import RequestContext
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import mcp_client, peers
from .cards import orchestrator_card
from .common import (BearerAuth, bind_request, configure_logging, env_url,
                     get_run_id, log_event, timed, unauthorized)
from .peers import PeerError
from .pooling import DEFAULT_W, SourceResult, reconcile
from .serve import JsonAgentExecutor, build_app, run

AGENT_NAME = "truthclf-orchestrator"

ZERO_SHOT_URL = env_url("ZERO_SHOT_AGENT_URL", "http://127.0.0.1:9101")
FINE_TUNED_URL = env_url("FINE_TUNED_AGENT_URL", "http://127.0.0.1:9102")
EXPLAINER_URL = env_url("EXPLAINER_AGENT_URL", "http://127.0.0.1:9103")
# Metric computation only. There is deliberately no model-tools address here.
DATA_TOOLS_URL = env_url("DATA_TOOLS_URL", "http://127.0.0.1:8081/mcp")

PREDICTOR_TIMEOUT = float(os.environ.get("PREDICTOR_TIMEOUT_S", 120))
EXPLAINER_TIMEOUT = float(os.environ.get("EXPLAINER_TIMEOUT_S", 600))
POOL_WEIGHT = float(os.environ.get("POOL_WEIGHT", DEFAULT_W))
MAX_POINTS = int(os.environ.get("MAX_POINTS", 50))

logger = logging.getLogger(AGENT_NAME)

# Peers, discovered once at start-up and shared by every request.
PEERS: dict[str, peers.Peer] = {}


# ---------------------------------------------------------------------------
# Reading agent replies
# ---------------------------------------------------------------------------
# A2A data parts are protobuf Struct values, in which every number is a double.
# Integers therefore arrive as floats -- a prediction of 1 comes back as 1.0 --
# so anything used as an integer or an identifier is coerced explicitly rather
# than trusted to have survived the hop.
def _as_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _index_predictions(payload: dict) -> dict[int, dict]:
    """Predictions keyed by row_id.

    Keyed rather than positional: a reply that omits or reorders rows would
    otherwise attach one statement's probability to another statement's label,
    and nothing downstream could detect it.
    """
    return {_as_int(p.get("row_id"), -1): p
            for p in (payload.get("predictions") or [])}


def _sources_from(payload: dict, row_ids: list[int], *, peer: str,
                  error: PeerError | None = None) -> dict[int, SourceResult]:
    """One SourceResult per requested row, whatever happened."""
    if error is not None:
        return {rid: SourceResult(status=error.status, reason=type(error).__name__,
                                  detail=error.detail) for rid in row_ids}
    if payload.get("status") == "unavailable":
        return {rid: SourceResult(status="unavailable",
                                  reason=payload.get("reason", "unavailable"),
                                  detail=payload.get("detail", ""))
                for rid in row_ids}

    indexed = _index_predictions(payload)
    # Rows the agent explicitly reported it could not score, as distinct from
    # rows simply absent from the reply. Both end up unusable, but the caller is
    # told which of the two happened.
    declared_missing = {_as_int(v, -1) for v in (payload.get("missing_row_ids") or [])}
    out = {}
    for rid in row_ids:
        hit = indexed.get(rid)
        if rid in declared_missing:
            out[rid] = SourceResult(
                status="unavailable", reason="FineTunedRowNotCached",
                detail=f"{peer} holds no recorded probability for this statement")
        elif hit is None:
            # The agent answered but this row is not in the reply.
            out[rid] = SourceResult(
                status="not_returned", reason="not_returned",
                detail=f"{peer} returned {len(indexed)} of {len(row_ids)} rows")
        else:
            out[rid] = SourceResult(status="ok",
                                    probability=float(hit["prob"]),
                                    prediction=_as_int(hit["pred"]))
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
async def verify(payload: dict, *, context_id: str) -> dict:
    """Run one verification: fan out, reconcile, explain, score."""
    points = payload.get("points") or []
    labels = payload.get("labels")
    if not points:
        return {"results": [], "metrics": None,
                "run": {"n": 0, "degraded": False, "warnings": [],
                        "run_id": get_run_id(), "context_id": context_id}}
    if len(points) > MAX_POINTS:
        raise ValueError(
            f"{len(points)} points exceeds the ceiling of {MAX_POINTS}. The "
            "request is refused rather than truncated: metrics over a shortened "
            "set answer a different question than the one asked.")
    if labels is not None and len(labels) != len(points):
        raise ValueError(f"{len(labels)} labels for {len(points)} points")

    # A stable identifier per statement, assigned here and used to join every
    # reply. A caller-supplied row_id is preserved so results can be matched
    # against the caller's own records.
    prepared = []
    for i, point in enumerate(points):
        item = dict(point)
        item["row_id"] = _as_int(item.get("row_id"), i) if "row_id" in item else i
        prepared.append(item)
    row_ids = [p["row_id"] for p in prepared]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique within a request")

    warnings: list[str] = []
    request = {"points": prepared}

    # --- fan out to the two predictors, concurrently ------------------------
    with timed() as fan:
        replies = await asyncio.gather(
            _ask("zero_shot", request, context_id, PREDICTOR_TIMEOUT),
            _ask("fine_tuned", request, context_id, PREDICTOR_TIMEOUT))
    (zs_payload, zs_error), (ft_payload, ft_error) = replies

    zs = _sources_from(zs_payload, row_ids, peer="zero-shot", error=zs_error)
    ft = _sources_from(ft_payload, row_ids, peer="fine-tuned", error=ft_error)

    for name, err in (("zero-shot", zs_error), ("fine-tuned", ft_error)):
        if err is not None:
            warnings.append(f"the {name} predictor did not answer ({err.status}): "
                            f"{err.detail[:200]}")
    n_uncovered = sum(1 for s in ft.values() if s.status == "unavailable")
    if n_uncovered:
        # States only what this predictor did. Naming the survivor here would be
        # a guess: when the other predictor has also failed, those statements get
        # no verdict at all rather than a zero-shot one.
        warnings.append(
            f"the fine-tuned predictor has no recorded probability for "
            f"{n_uncovered} of {len(row_ids)} statement(s); reconciliation did "
            "not run for those")

    # --- reconcile ----------------------------------------------------------
    results = [reconcile(zs[rid], ft[rid], w=POOL_WEIGHT) for rid in row_ids]
    for point, result in zip(prepared, results):
        result["row_id"] = point["row_id"]
        result["statement"] = point.get("statement", "")

    log_event(logger, "fan-out complete", n=len(prepared), duration_ms=fan.ms,
              pooled=sum(1 for r in results if r["reconciliation"]["applied"]),
              single_source=sum(1 for r in results
                                if r["reconciliation"]["reason"] == "single_source"),
              no_verdict=sum(1 for r in results if r["status"] == "no_verdict"))

    # --- explain, after the verdict is settled ------------------------------
    explanations, exp_error = await _ask(
        "explainer", {"points": prepared, "with_rationale": True},
        context_id, EXPLAINER_TIMEOUT)
    if exp_error is not None:
        warnings.append(f"the explainer did not answer ({exp_error.status}): "
                        f"{exp_error.detail[:200]}; verdicts are unaffected")
        by_row: dict[int, dict] = {}
    else:
        by_row = {_as_int(pp.get("row_id"), -1): pp
                  for pp in (explanations.get("per_point") or [])}

    for result in results:
        pp = by_row.get(result["row_id"])
        result["explanation"] = None if pp is None else {
            "driver": pp.get("driver"),
            "rationale": pp.get("rationale"),
            "occlusion": pp.get("occlusion"),
            "rationale_refs": pp.get("rationale_refs"),
            "rationale_agrees_with_driver": pp.get("agree"),
            "note": ("The driver is measured by removing each field and observing "
                     "the shift in predicted probability. The rationale is the "
                     "model's own account and is not necessarily faithful; agreement "
                     "between the two is not distinguishable from chance."),
        }

    # --- metrics, when the caller supplied labels ---------------------------
    metrics = None
    if labels is not None:
        metrics = await _metrics(results, labels, row_ids, warnings)

    degraded = bool(warnings) or any(r["status"] == "no_verdict" for r in results)
    return {
        "results": results,
        "metrics": metrics,
        "run": {
            "n": len(results),
            "degraded": degraded,
            "warnings": warnings,
            "run_id": get_run_id(),
            "context_id": context_id,
            "pooling": {"method": "log_odds_pool", "w": POOL_WEIGHT},
        },
    }


async def _ask(peer_key: str, payload: dict, context_id: str, timeout: float):
    """Call one peer. Returns (payload, error); never raises for a peer failure.

    A predictor that fails is a degraded run, not a failed one: the other
    predictor's answer is still a verdict, and the caller asked about the
    statements rather than about the agents.
    """
    peer = PEERS.get(peer_key)
    if peer is None:
        # Attempt lazy discovery if not yet found in PEERS
        url = {
            "zero_shot": ZERO_SHOT_URL,
            "fine_tuned": FINE_TUNED_URL,
            "explainer": EXPLAINER_URL
        }.get(peer_key)
        if url:
            try:
                log_event(logger, "attempting lazy peer discovery", peer=peer_key, url=url)
                peer = await peers.discover(peer_key, url)
                PEERS[peer_key] = peer
            except Exception as e:
                return {}, PeerError(peer_key, "error", f"lazy discovery failed: {e}")
        else:
            return {}, PeerError(peer_key, "error", f"unknown peer key: {peer_key}")

    try:
        reply = await peer.ask(payload, context_id=context_id, timeout=timeout)
    except PeerError as e:
        return {}, e
    return reply.payload, None


async def _metrics(results: list[dict], labels: list, row_ids: list[int],
                   warnings: list[str]) -> dict:
    """Score the run through the metrics tool.

    Only statements that have a verdict can be scored. Both counts are reported,
    because an accuracy over a silently reduced subset is a different quantity
    from the one the caller asked for.
    """
    by_row = dict(zip(row_ids, zip(results, labels)))
    scored_y, scored_pred, scored_prob = [], [], []
    for rid in row_ids:
        result, label = by_row[rid]
        if result["status"] != "ok":
            continue
        scored_y.append(label)
        scored_pred.append(1 if result["verdict"] else 0)
        scored_prob.append(result["probability"])

    excluded = len(results) - len(scored_y)
    if excluded:
        warnings.append(f"{excluded} statement(s) had no verdict and are excluded "
                        "from the metrics")
    if not scored_y:
        return {"n_scored": 0, "n_excluded": excluded, "reconciled": None,
                "note": "no statement received a verdict, so nothing could be scored"}

    payload = await mcp_client.call_tool(DATA_TOOLS_URL, "metrics", {
        "y_true": scored_y, "preds": scored_pred, "probs": scored_prob,
        "bootstrap": True, "n_boot": 1000, "seed": 0})
    return {
        "n_scored": len(scored_y),
        "n_excluded": excluded,
        "reconciled": payload["metrics"],
        "baseline": payload["baseline"],
        "warnings": payload.get("warnings", []),
        "note": ("Metrics are computed over the statements that received a verdict. "
                 "Every figure carries a bootstrap interval, and the ECE is reported "
                 "with the number of bins it was computed from."),
    }


# ---------------------------------------------------------------------------
# A2A executor and the public HTTP entry point
# ---------------------------------------------------------------------------
class OrchestratorExecutor(JsonAgentExecutor):
    agent_name = AGENT_NAME

    async def handle(self, payload: dict, context: RequestContext) -> tuple[dict, str]:
        result = await verify(payload, context_id=context.context_id)
        run = result["run"]
        return result, (f"Verified {run['n']} statement(s); "
                        f"{'degraded' if run['degraded'] else 'all sources answered'}.")


def verify_route(auth: BearerAuth):
    """POST /verify -- the public entry point.

    Plain HTTP by design: the caller is a client, not an agent, and A2A governs
    what agents say to each other. The request is turned into the same run the
    A2A skill performs, so the two entry points cannot drift.
    """

    async def handler(request: Request) -> JSONResponse:
        if not auth.check(request):
            return unauthorized()
        bind_request(traceparent=request.headers.get("traceparent", ""),
                     run_id=request.headers.get("x-truthclf-run-id", ""))
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "InvalidJSON",
                                 "detail": "request body must be a JSON object"},
                                status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "InvalidJSON",
                                 "detail": "request body must be a JSON object"},
                                status_code=400)
        context_id = payload.get("context_id") or uuid.uuid4().hex
        log_event(logger, "verify requested", n=len(payload.get("points") or []),
                  has_labels=payload.get("labels") is not None)
        try:
            result = await verify(payload, context_id=context_id)
        except ValueError as e:
            # A malformed request, distinct from a run that could not complete.
            return JSONResponse({"error": type(e).__name__, "detail": str(e)},
                                status_code=400)
        return JSONResponse(result,
                            headers={"x-truthclf-run-id": get_run_id()})

    return handler


def main() -> None:
    ap = argparse.ArgumentParser(description=AGENT_NAME)
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 9100)))
    args = ap.parse_args()

    configure_logging(AGENT_NAME)
    # The public token for the /verify endpoint. In GCP, this is checked by the
    # A2A Gateway and the app checks it again as defense-in-depth.
    # Agent-to-agent calls will be authenticated by per-hop OIDC ID tokens
    # validated by Cloud Run's IAM layer, so no app-level token is needed there.
    public_auth = BearerAuth("ORCHESTRATOR_TOKEN")

    async def startup():
        async def discover_loop():
            # Discover peers in background
            missing_peers = {
                "zero_shot": ZERO_SHOT_URL,
                "fine_tuned": FINE_TUNED_URL,
                "explainer": EXPLAINER_URL
            }
            delay = 1.0
            while missing_peers:
                # If a peer was already discovered by a lazy load, we skip it
                for key in list(missing_peers.keys()):
                    if key in PEERS:
                        del missing_peers[key]

                if not missing_peers:
                    break

                to_remove = []
                for key, url in list(missing_peers.items()):
                    try:
                        PEERS[key] = await peers.discover(key, url)
                        to_remove.append(key)
                    except Exception as e:
                        log_event(logger, "peer discovery failed, retrying...", peer=key, url=url,
                                  error=type(e).__name__, detail=str(e)[:150])
                for key in to_remove:
                    del missing_peers[key]
                if missing_peers:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

            # Probe data-tools in background
            data_tools_connected = False
            delay = 1.0
            while not data_tools_connected:
                try:
                    tools = await mcp_client.probe(DATA_TOOLS_URL)
                    log_event(logger, "connected to data-tools", url=DATA_TOOLS_URL, tools=tools)
                    data_tools_connected = True
                except Exception as e:
                    log_event(logger, "data-tools probe failed, retrying...", url=DATA_TOOLS_URL,
                              error=type(e).__name__, detail=str(e)[:150])
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

        asyncio.create_task(discover_loop())

    app = build_app(
        agent_name=AGENT_NAME, executor=OrchestratorExecutor(),
        card_builder=orchestrator_card, auth=None,
        extra_routes=[Route("/verify", verify_route(public_auth), methods=["POST"])],
        on_startup=startup)
    run(app, args.host, args.port, AGENT_NAME)


if __name__ == "__main__":
    main()

"""MCP server exposing prediction and explanation.

This server owns the provider credential, the on-disk response cache and the
calibrator artifacts. It is the only one of the two that reaches the network.

Both predictors are exposed through a single `predict` tool with a required
`model` argument rather than as two tools. They are interchangeable by design --
same inputs, same outputs -- and one tool with a discriminator keeps that
property enforced instead of letting two response shapes drift apart. The
argument has no default: a caller that meant the fine-tuned model must not
silently receive zero-shot results.

Run:  python -m truthclf_mcp.model_tools [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import logging
import math
import os
import warnings
from typing import Annotated, Literal

import numpy as np
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from truthclf import explain as X
from truthclf import llm, metrics as M, prompts
from truthclf.evaluation import DecisionArtifact
from truthclf.predictors import FinetunedPredictor, ZeroShotPredictor
from truthclf.predictors import finetuned as FT

from .adapter import (MAX_EXPLAIN_POINTS, MAX_PREDICT_POINTS, LabelValue,
                      Point, Scheme, check_batch, prepare)
from .errors import (CounterfactualNotAvailable, FineTunedModelNotServable,
                     FineTunedRowNotCached, ProviderCredentialMissing,
                     tool_errors)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.environ.get(
    "TRUTHCLF_CALIBRATOR_DIR", os.path.join(PROJECT_ROOT, "results/calibrators"))

ZERO_SHOT_MODEL = os.environ.get("TRUTHCLF_ZEROSHOT_MODEL", "google/gemma-4-31B-it")
FINE_TUNED_MODEL = os.environ.get(
    "TRUTHCLF_FT_MODEL",
    "makisntpap_17e5/gemma-4-31B-it-gemma_truth_sft-c7afbf0d")
FT_PROB_CACHE = os.environ.get(
    "TRUTHCLF_FT_CACHE", os.path.join(PROJECT_ROOT, FT.FT_PROB_CACHE))
# Companion to the probabilities: which statement each one was computed for.
# Built by scripts/build_ft_identity.py.
FT_IDENTITY = os.environ.get(
    "TRUTHCLF_FT_IDENTITY", os.path.join(PROJECT_ROOT, "ft_eval_identity.json"))

# Measured on a batch refetch of 6,075 rows: about 150 input tokens per row at
# the base model's published rate. Used only for the pre-flight cost estimate.
_OUTPUT_TOKENS_PER_CALL = {"logprob": 1, "score": 4}

server = MCPServer(
    name="truthclf-model-tools",
    instructions="Truthfulness prediction and field-occlusion explanation. "
                 "Holds the provider credential and the fitted calibrators. "
                 "`predict` requires an explicit model choice; the fine-tuned "
                 "model is served from stored probabilities and says so in "
                 "every response.",
)


# ---------------------------------------------------------------------------
# Calibrator registry, loaded once at start-up
# ---------------------------------------------------------------------------
class _Calibrators:
    """Every artifact in the calibrator directory, keyed by (model, elicitation).

    Loaded at import rather than per request. The files are a few hundred bytes
    each, so this is not about cost: a missing, stale-schema or mismatched
    artifact is a deployment fault, and discovering it at start-up is better
    than discovering it on whichever request happens to arrive first.

    The key needs both parts. A model id alone does not identify a probability
    scale -- the two zero-shot artifacts share one -- so selecting on the model
    would silently accept the score-mode calibrator for logprob probabilities.
    The artifact's own check_model is a second line of defence, not the
    selector.
    """

    def __init__(self, directory: str):
        self.directory = directory
        self.by_key: dict[tuple[str, str], DecisionArtifact] = {}
        self.paths: dict[tuple[str, str], str] = {}
        for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
            art = DecisionArtifact.load(path)
            key = (art.model, art.elicitation)
            if key in self.by_key:
                raise RuntimeError(
                    f"two calibrators claim {key}: {self.paths[key]} and {path}. "
                    "One probability scale cannot have two mappings.")
            self.by_key[key] = art
            self.paths[key] = path

    def get(self, model: str, elicitation: str) -> DecisionArtifact:
        key = (model, elicitation)
        if key not in self.by_key:
            raise FileNotFoundError(
                f"no calibrator for model={model!r} elicitation={elicitation!r} "
                f"in {self.directory}. Available: {sorted(self.by_key)}")
        return self.by_key[key]

    def describe(self, model: str, elicitation: str) -> dict:
        art = self.get(model, elicitation)
        return {"artifact": os.path.basename(self.paths[(model, elicitation)]),
                "method": art.calibrator.get("method"),
                "threshold": art.threshold,
                "objective": art.objective,
                "fitted_on": art.fitted_on,
                "n_val": art.n_val}


CALIBRATORS = _Calibrators(ARTIFACT_DIR)

# The credential is checked but not required to start: the stored fine-tuned
# path and the cost-estimate path are fully functional without one, so refusing
# to boot would disable working tools. A live call raises instead.
_HAS_KEY = bool(os.environ.get("TOGETHER_API_KEY"))


def _require_key() -> None:
    if not os.environ.get("TOGETHER_API_KEY"):
        raise ProviderCredentialMissing(
            "TOGETHER_API_KEY is not set, so no live model call can be made. "
            "Stored fine-tuned probabilities and estimate_only=true still work.")


def _clean_json(x):
    """Convert NaN to None for JSON output.

    JSON has no NaN. Encoding it as 0.0 would make an undefined metric
    indistinguishable from a genuinely poor one.
    """
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) else f
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def _estimate(rows, elicitation: str, model: str) -> dict:
    """Token and cost estimate for calling `model` on these rows, before calling."""
    mode = "decision" if elicitation == "logprob" else "score"
    n_in = sum(llm.count_tokens(m["content"])
               for r in rows for m in prompts.build_messages(r, "full", mode=mode))
    n_out = len(rows) * _OUTPUT_TOKENS_PER_CALL[elicitation]
    try:
        usd = llm.estimate_cost(n_in, n_out, model)
    except KeyError:
        # Fine-tuned model ids are not in the published price table.
        usd = None
    return {"input_tokens": n_in, "output_tokens": n_out, "usd": usd,
            "basis": f"{llm.TOKENIZER}; cache hits are not charged and are not "
                     "subtracted from this figure"}


def _metrics_with_bins(y, preds, probs) -> dict:
    bundle = M.metric_bundle(y, preds, probs)
    bundle["ece_bins"] = M.ece_bin_report(y, probs)
    return bundle


# ---------------------------------------------------------------------------
# Fine-tuned serving
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _stored_identity() -> dict[int, str]:
    """row_id -> the normalised statement key each stored probability belongs to.

    A row_id is a position in the source dataset, not a statement. Without this
    map the serving path cannot tell whether the probability it is about to
    return was computed for the statement being asked about, and a caller
    supplying its own text under an existing row_id gets another statement's
    probability, correctly calibrated and labelled as a successful prediction.

    Built by scripts/build_ft_identity.py and cached: the file is written once
    beside the probabilities and does not change while the server runs.
    """
    import json as _json
    if not os.path.exists(FT_IDENTITY):
        raise FileNotFoundError(
            f"{FT_IDENTITY} not found. Stored fine-tuned probabilities cannot be "
            "served without it, because a row_id alone does not identify a "
            "statement. Build it with scripts/build_ft_identity.py.")
    with open(FT_IDENTITY, encoding="utf-8") as f:
        return {int(k): v for k, v in _json.load(f)["identity"].items()}


def _stored_row_ids() -> frozenset[int]:
    """Row ids the recorded fine-tuned probabilities cover."""
    return frozenset(_stored_identity())


def _partition_by_coverage(rows):
    """Split rows by whether a stored probability exists FOR THAT STATEMENT.

    Both keys are checked. Matching the row_id alone is what allowed another
    statement's probability to be served; matching the normalised key alone
    would lose the binding to the recorded evaluation. A row_id that matches
    while the statement does not is the dangerous case, so it is separated from
    a plain absence and reported.
    """
    identity = _stored_identity()
    covered, absent, mismatched = [], [], []
    for r in rows:
        expected = identity.get(r.row_id)
        if expected is None:
            absent.append(r.row_id)
        elif expected != r.norm_key:
            mismatched.append(r.row_id)
        else:
            covered.append(r)
    if mismatched:
        # Normalisation is exact-match after lowering and punctuation removal,
        # and it is known to over-merge occasionally, so equality is a strong
        # signal of identity but not a proof of it. A mismatch, by contrast, is
        # conclusive: the statement asked about is not the statement scored.
        logging.getLogger(__name__).warning(
            "fine-tuned identity mismatch on row_ids %s: a stored probability "
            "exists for these row_ids but was computed for a different "
            "statement. Refusing to serve it.", mismatched[:10])
    return covered, absent, mismatched


def _verify_identity(rows):
    """Refuse rows whose stored probability belongs to a different statement."""
    _, absent, mismatched = _partition_by_coverage(rows)
    if mismatched:
        raise FineTunedRowNotCached(
            f"{len(mismatched)} row_id(s) have a stored probability that was "
            f"computed for a DIFFERENT statement (first few: {mismatched[:5]}). "
            "A row_id is a position in the source dataset, not a statement, so "
            "the stored value is not an answer to this question and is not "
            "served.")
    if absent:
        raise FineTunedRowNotCached(
            f"{len(absent)} statement(s) have no stored fine-tuned probability "
            f"(first few row_ids: {absent[:5]}). Stored probabilities cover the "
            "statements evaluated during the endpoint session; scoring others "
            "requires a live endpoint.")


def _finetuned_stored(rows):
    """Per-row fine-tuned probabilities from the recorded evaluation session.

    The adapter cannot be served without a dedicated endpoint, so its
    probabilities are replayed from the file written when one was last running.
    Values there are raw model output; calibration is applied by the caller,
    exactly as the evaluation pipeline does.

    A row with no stored probability is an error. It never falls back to a live
    call and never falls back to the zero-shot predictor.
    """
    _verify_identity(rows)
    try:
        return FT.load_cached_probs(rows, FT_PROB_CACHE)
    except FileNotFoundError as e:
        raise FineTunedRowNotCached(str(e)) from e
    except KeyError as e:
        raise FineTunedRowNotCached(
            f"{e.args[0] if e.args else e} -- stored probabilities cover only "
            "the rows evaluated during the endpoint session (the validation and "
            "test splits). Scoring other rows requires a live endpoint.") from e


def _build_predictor(model: str, elicitation: str, calibrator=None):
    """Construct the predictor for a live call.

    The fine-tuned choice yields a FinetunedPredictor rather than a zero-shot
    predictor aimed at the fine-tuned model id. The two currently behave the
    same, since FinetunedPredictor delegates to a zero-shot predictor bound to
    its served model, but the serving configuration for the fine-tuned model
    belongs on the type that owns it: when it is served from somewhere with a
    live endpoint, that is where the change lands.
    """
    _require_key()
    served = ZERO_SHOT_MODEL if model == "zero_shot" else FINE_TUNED_MODEL
    client = llm.make_client(served)
    use_logprobs = elicitation == "logprob"
    if model == "fine_tuned":
        return FinetunedPredictor(
            base_model=ZERO_SHOT_MODEL, served_model=served, variant="full",
            client=client, use_logprobs=use_logprobs, calibrator=calibrator)
    return ZeroShotPredictor(
        model=served, variant="full", client=client,
        use_logprobs=use_logprobs, calibrator=calibrator)


@contextlib.contextmanager
def _live_call(model: str):
    """Translate a provider refusal to serve the fine-tuned model.

    Wraps every live call so the same underlying failure reports identically
    whichever tool the caller used. The request is issued rather than
    pre-empted, so what is reported is the provider's current answer and not an
    assumption recorded in this file.
    """
    try:
        yield
    except Exception as e:
        text = str(e)
        if model == "fine_tuned" and (
                "model_not_available" in text or "non-serverless" in text):
            raise FineTunedModelNotServable(
                f"the provider refused to serve {FINE_TUNED_MODEL}: {e}. This is "
                "a capability response, not a transient failure, and is not "
                "retried. Use fine_tuned_source='cached' to replay the recorded "
                "evaluation, or serve the model from a live endpoint.") from e
        raise


# ---------------------------------------------------------------------------
# 1. predict
# ---------------------------------------------------------------------------
@server.tool(
    title="Truthfulness prediction",
    description="Predict True/False for a set of statements, with calibrated "
                "probabilities. Returns evaluation metrics when labels are "
                "supplied. `model` must be given explicitly.",
)
@tool_errors
def predict(
    model: Annotated[Literal["zero_shot", "fine_tuned"], Field(
        description="Which predictor to use. Required: there is no default, so "
                    "a caller cannot receive zero-shot results by omission.")],
    points: Annotated[list[Point], Field(
        description=f"Statements to score. Ceiling {MAX_PREDICT_POINTS} per call.")],
    labels: list[LabelValue] | None = None,
    scheme: Scheme = "primary",
    elicitation: Annotated[Literal["logprob", "score"], Field(
        description="logprob is the reported baseline; score is the continuous "
                    "variant used by the explainer.")] = "logprob",
    calibrated: bool = True,
    fine_tuned_source: Annotated[Literal["cached", "live"], Field(
        description="Ignored when model='zero_shot'. 'cached' replays the "
                    "probabilities recorded during the endpoint evaluation.")] = "cached",
    estimate_only: Annotated[bool, Field(
        description="Return the token and cost estimate without calling the "
                    "model.")] = False,
    max_live_calls: Annotated[int, Field(
        ge=0,
        description="Budget ceiling on uncached provider calls. Refuses rather "
                    "than exceeding. This is a spend limit, not an approval."
    )] = MAX_PREDICT_POINTS,
    on_missing: Annotated[Literal["error", "omit"], Field(
        description="For the stored fine-tuned path, what to do about statements "
                    "with no recorded probability. 'error' refuses the batch; "
                    "'omit' scores the rest and lists the ones it could not, in "
                    "missing_row_ids. Neither substitutes another predictor."
    )] = "error",
) -> dict:
    """Score a set of statements and, given labels, evaluate the predictions."""
    check_batch(len(points), MAX_PREDICT_POINTS, "points", "predict")
    rows, y = prepare(points, labels, scheme)
    served_model = ZERO_SHOT_MODEL if model == "zero_shot" else FINE_TUNED_MODEL
    warns: list[str] = []

    if not rows:
        # An empty batch is a defined no-op for a service. There is nothing to
        # score, so metrics stay null rather than raising from inside sklearn.
        return _clean_json({
            "n": 0, "predictions": [], "threshold": None, "parse_failures": 0,
            "metrics": None,
            "provenance": {"model_id": served_model, "served_by": "none",
                           "elicitation": elicitation, "calibrated": calibrated,
                           "calibrator": None, "cache_hits": 0, "live_calls": 0,
                           "cached_session": None},
            "estimate": None, "warnings": warns})

    artifact = CALIBRATORS.get(served_model, elicitation) if calibrated else None

    estimate = _estimate(rows, elicitation, served_model)
    if estimate_only:
        return _clean_json({
            "n": len(rows), "predictions": [], "threshold": None,
            "parse_failures": 0, "metrics": None,
            "provenance": {"model_id": served_model, "served_by": "none",
                           "elicitation": elicitation, "calibrated": calibrated,
                           "calibrator": (CALIBRATORS.describe(served_model, elicitation)
                                          if artifact else None),
                           "cache_hits": 0, "live_calls": 0, "cached_session": None},
            "estimate": estimate, "warnings": warns})

    missing_row_ids: list[int] = []
    if model == "fine_tuned" and fine_tuned_source == "cached":
        if elicitation != "logprob":
            raise ValueError(
                "stored fine-tuned probabilities were produced by single-token "
                "logprob elicitation; serving them as elicitation='score' would "
                "mislabel their scale")
        if on_missing == "omit":
            # Coverage is partial by nature, so a batch mixing recorded and
            # unrecorded statements scores what it can and names the rest. The
            # omitted rows are still never answered by another predictor.
            kept, absent, mismatched = _partition_by_coverage(rows)
            missing_row_ids = sorted(absent + mismatched)
            if mismatched:
                warns.append(
                    f"{len(mismatched)} row_id(s) have a stored probability "
                    "computed for a different statement and were refused, not "
                    f"served: {mismatched[:5]}.")
            if missing_row_ids and y is not None:
                # Labels must be filtered with the rows they belong to, or the
                # metrics would be computed against another statement's truth.
                keep = {r.row_id for r in kept}
                y = [label for r, label in zip(rows, y) if r.row_id in keep]
            rows = kept
            if missing_row_ids:
                warns.append(
                    f"{len(missing_row_ids)} of "
                    f"{len(rows) + len(missing_row_ids)} statement(s) have no "
                    "recorded fine-tuned probability and were omitted; their "
                    "row_ids are in missing_row_ids. They were not scored by "
                    "any other model.")
        raw = [float(p) for p in _finetuned_stored(rows)] if rows else []
        served_by = "cached_replay"
        parse_failures = 0
        cache_hits, live_calls = len(rows), 0
        cached_session = (f"stored per-row probabilities from the dedicated "
                          f"endpoint evaluation ({os.path.basename(FT_PROB_CACHE)})")
    else:
        _require_key()
        if len(rows) > max_live_calls:
            raise ValueError(
                f"{len(rows)} points would exceed max_live_calls={max_live_calls}. "
                "No call was made. Raise the ceiling or reduce the batch; "
                f"estimated cost of the full batch: {estimate}")
        # Calibration is applied below from the registry artifact, uniformly for
        # the live and stored paths, so the predictor is built without one.
        predictor = _build_predictor(model, elicitation)
        with _live_call(model):
            result = predictor.predict(rows)
        raw = [float(p) for p in result.probs]
        parse_failures = result.parse_failures
        served_by = "live"
        cache_hits, live_calls = 0, len(rows)
        cached_session = None
        if parse_failures:
            warns.append(
                f"{parse_failures}/{len(rows)} responses did not yield a usable "
                "probability and fell back to a neutral 0.5. Those are absent "
                "measurements, not measurements of 0.5.")

    if artifact is not None:
        probs, preds = artifact.decide(raw)
        threshold = artifact.threshold
    else:
        probs, preds = raw, [1 if p >= 0.5 else 0 for p in raw]
        threshold = 0.5
        warns.append(
            "calibrated=false: probabilities are raw model output decided at "
            "0.5. Raw probabilities from this model are badly calibrated "
            "(expected calibration error around 0.32 against 0.06 calibrated), "
            "and the fitted threshold is not 0.5.")

    predictions = [
        {"row_id": r.row_id, "pred": int(pr), "prob": float(p),
         "prob_raw": (float(rw) if artifact is not None else None), "score": None}
        for r, p, pr, rw in zip(rows, probs, preds, raw)]

    return _clean_json({
        "n": len(rows),
        "predictions": predictions,
        # Statements this call could not score. Empty unless on_missing="omit".
        "missing_row_ids": missing_row_ids,
        "threshold": threshold,
        "parse_failures": parse_failures,
        "metrics": (_metrics_with_bins(y, preds, probs)
                    if y is not None and rows else None),
        "provenance": {
            "model_id": served_model,
            "served_by": served_by,
            "elicitation": elicitation,
            "calibrated": artifact is not None,
            "calibrator": (CALIBRATORS.describe(served_model, elicitation)
                           if artifact else None),
            "cache_hits": cache_hits, "live_calls": live_calls,
            "cached_session": cached_session,
        },
        "estimate": estimate,
        "warnings": warns,
    })


# ---------------------------------------------------------------------------
# 2. explain
# ---------------------------------------------------------------------------
_AGREEMENT_NOTE = (
    "The rationale-to-driver agreement rate is not distinguishable from chance: "
    "a permutation null over shuffled driver labels gives 0.436 with a 95% range "
    "of [0.393, 0.480] against an observed 0.457, p = 0.19. Read it as evidence "
    "that the rationales carry no detectable information about which field drove "
    "the prediction, not as a measured faithfulness rate.")


@server.tool(
    title="Prediction explanation",
    description="Explain predictions by leave-one-field-out occlusion, with the "
                "model's own rationale and a cross-check between the two. "
                "Returns metrics when labels are supplied.",
)
@tool_errors
def explain(
    points: Annotated[list[Point], Field(
        description=f"Statements to explain. Ceiling {MAX_EXPLAIN_POINTS} per "
                    "call, because each point costs six occlusion calls plus "
                    "one rationale call.")],
    model: Annotated[Literal["zero_shot", "fine_tuned"], Field(
        description="Which predictor to explain. Defaults to zero_shot because "
                    "it is the only one currently explainable: the fine-tuned "
                    "model has no live endpoint, and its stored probabilities "
                    "cannot answer occlusion queries. This reflects serving "
                    "availability, not a judgement about which predictor is "
                    "more worth explaining. fine_tuned remains requestable and "
                    "fails with the specific reason.")] = "zero_shot",
    labels: list[LabelValue] | None = None,
    scheme: Scheme = "primary",
    with_rationale: bool = True,
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    driver_eps: Annotated[float, Field(
        ge=0.0, le=1.0,
        description="A field counts as the driver only if removing it moves the "
                    "probability by more than this.")] = 0.05,
    max_parse_failure_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
    elicitation: Annotated[Literal["score", "logprob"], Field(
        description="Occlusion needs graded probabilities; the logprob path is "
                    "near-saturated and its deltas collapse to 0/1 jumps.")] = "score",
    calibrated: bool = False,
    fine_tuned_source: Literal["cached", "live"] = "cached",
) -> dict:
    """Attribute each prediction to an input field and cross-check the rationale."""
    check_batch(len(points), MAX_EXPLAIN_POINTS, "points", "explain",
                note=f"Each point issues 6 occlusion calls plus a rationale call, "
                     f"so the ceiling bounds provider requests at "
                     f"{MAX_EXPLAIN_POINTS * 7}. ")
    rows, y = prepare(points, labels, scheme)
    served_model = ZERO_SHOT_MODEL if model == "zero_shot" else FINE_TUNED_MODEL
    warns: list[str] = []

    if not rows:
        return _clean_json({
            "n": 0, "per_point": [],
            "aggregate": {"field_table": [], "driver_distribution": {},
                          "rationale_occlusion_agreement_rate": None, "n": 0},
            "parse_failures": 0, "parse_failure_rate": None, "n_predictions": 0,
            "metrics": None,
            "provenance": {"model_id": served_model, "served_by": "none",
                           "elicitation": elicitation, "calibrated": calibrated,
                           "driver_eps": driver_eps, "threshold": threshold},
            "warnings": warns, "interpretation_note": _AGREEMENT_NOTE})

    if model == "fine_tuned" and fine_tuned_source == "cached":
        raise CounterfactualNotAvailable(
            "the fine-tuned model has no live endpoint, and its stored "
            "probabilities are keyed on row_id alone. Occlusion builds "
            "field-ablated copies that keep the same row_id, so every ablated "
            "variant would return the base row's probability: all deltas would "
            "be zero, every point would be attributed to the statement, and the "
            "resulting driver distribution would be an artifact of the join key "
            "rather than a measurement. Explain the zero-shot predictor, or "
            "provision an endpoint and pass fine_tuned_source='live'.")

    artifact = CALIBRATORS.get(served_model, elicitation) if calibrated else None
    if artifact is not None:
        warns.append(
            f"calibrated=true: probabilities are calibrated, but this tool's "
            f"threshold ({threshold}) is independent of the calibrator's fitted "
            f"threshold ({artifact.threshold:.4f}), so predictions here are not "
            "the ones the calibrated predictor would make.")

    predictor = _build_predictor(model, elicitation, calibrator=artifact)
    with _live_call(model):
        result = X.explain(predictor, rows, labels=y,
                           with_rationale=with_rationale, threshold=threshold,
                           driver_eps=driver_eps,
                           max_parse_failure_rate=max_parse_failure_rate)
    agg = X.aggregate(result)
    # aggregate() returns the field table as a DataFrame; convert at the
    # boundary rather than shipping a non-serialisable object.
    agg["field_table"] = agg["field_table"].to_dict(orient="records")

    return _clean_json({
        "n": len(rows),
        "per_point": result["per_point"],
        "aggregate": agg,
        "parse_failures": result["parse_failures"],
        "parse_failure_rate": result["parse_failure_rate"],
        "n_predictions": result["n_predictions"],
        "metrics": result.get("metrics"),
        "provenance": {"model_id": served_model, "served_by": "live",
                       "elicitation": elicitation,
                       "calibrated": artifact is not None,
                       "driver_eps": driver_eps, "threshold": threshold},
        "warnings": warns,
        "interpretation_note": _AGREEMENT_NOTE,
    })


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
# Deliberately outside the MCP protocol and unauthenticated: a container
# orchestrator probes this before any client has credentials, and it is how
# dependent services learn this one is ready. It reports what was loaded rather
# than a bare "ok", so a half-initialised server cannot pass.
@server.custom_route("/healthz", methods=["GET"])
async def healthz(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "model-tools",
                         "calibrators": len(CALIBRATORS.by_key),
                         "stored_finetuned_rows": len(_stored_identity()),
                         "provider_credential": _HAS_KEY})


def main() -> None:
    ap = argparse.ArgumentParser(description="truthclf model-tools MCP server")
    ap.add_argument("--host", default=os.environ.get("TRUTHCLF_MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("TRUTHCLF_MODEL_TOOLS_PORT", 8082)))
    ap.add_argument("--path", default="/mcp")
    args = ap.parse_args()
    print(f"model-tools: {len(CALIBRATORS.by_key)} calibrators loaded "
          f"{sorted(CALIBRATORS.by_key)}", flush=True)
    if not _HAS_KEY:
        warnings.warn(
            "TOGETHER_API_KEY is not set. Stored fine-tuned probabilities and "
            "estimate_only=true work; any live call will raise.",
            RuntimeWarning, stacklevel=2)
    print(f"model-tools listening on http://{args.host}:{args.port}{args.path}",
          flush=True)
    server.run(transport="streamable-http", host=args.host, port=args.port,
               streamable_http_path=args.path)


if __name__ == "__main__":
    main()

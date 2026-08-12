"""Agent cards, and the route that serves them.

An agent card is how another agent discovers what this one does and where to
call it. It is built per request so the advertised address is the one the caller
actually reached, which no static configuration can guarantee.

Note on the types: in a2a-sdk 1.1.2 these are protobuf messages, not pydantic
models. They are built with keyword arguments; repeated fields are assigned with
a list, and map fields with a dict. There is no top-level `url` on AgentCard --
the address lives in `supported_interfaces`.
"""

from __future__ import annotations

from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from a2a.types import (AgentCapabilities, AgentCard, AgentInterface,
                       AgentProvider, AgentSkill, HTTPAuthSecurityScheme,
                       SecurityRequirement, SecurityScheme, StringList)
from a2a.utils import AGENT_CARD_WELL_KNOWN_PATH, TransportProtocol
from starlette.requests import Request
from starlette.responses import JSONResponse

from .common import resolve_base_url

PROTOCOL_VERSION = "1.0"
JSON_MODE = "application/json"
ORGANIZATION = "truthclf"

# One bearer scheme, named once, referenced from the card and from each skill.
# Repeating it at skill level means a reader of a single skill sees the
# requirement without having to resolve the whole card.
BEARER = "bearer"


def _security() -> tuple[dict, list]:
    schemes = {BEARER: SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(
            scheme="bearer",
            description="Shared bearer token, sent as: Authorization: Bearer <token>"))}
    requirements = [SecurityRequirement(schemes={BEARER: StringList(list=[])})]
    return schemes, requirements


def _card(*, name: str, description: str, base_url: str, streaming: bool,
          skill: AgentSkill, version: str = "1.0.0") -> AgentCard:
    schemes, requirements = _security()
    return AgentCard(
        name=name,
        description=description,
        version=version,
        provider=AgentProvider(organization=ORGANIZATION, url=base_url),
        supported_interfaces=[AgentInterface(
            url=base_url,
            protocol_binding=TransportProtocol.JSONRPC,
            protocol_version=PROTOCOL_VERSION)],
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=False,
            extended_agent_card=False),
        security_schemes=schemes,
        security_requirements=requirements,
        default_input_modes=[JSON_MODE],
        default_output_modes=[JSON_MODE],
        skills=[skill],
    )


def _skill(**kw) -> AgentSkill:
    _, requirements = _security()
    kw.setdefault("input_modes", [JSON_MODE])
    kw.setdefault("output_modes", [JSON_MODE])
    return AgentSkill(security_requirements=requirements, **kw)


# Every payload here is a batch of structured records -- statements,
# probabilities, per-field occlusion deltas -- so the machine-readable channel
# is always a JSON data part. A human-readable summary may ride alongside as
# text, but nothing downstream parses it.
_EXAMPLE = ('{"points": [{"statement": "Our state added 50,000 jobs last quarter.", '
            '"speaker_name": "Jane Doe", "speaker_affiliation": "democrat"}], '
            '"labels": ["mostly-true"]}')


def orchestrator_card(base_url: str) -> AgentCard:
    return _card(
        name="truthclf-orchestrator",
        description=(
            "Verifies the truthfulness of a batch of statements. Fans out to the "
            "zero-shot and fine-tuned predictor agents over A2A, reconciles their "
            "calibrated probabilities by log-odds pooling, obtains explanations "
            "from the explainer agent, and returns one verdict per statement. "
            "Returns aggregate performance metrics when the caller supplies labels."),
        base_url=base_url,
        # Advertised so an A2A caller may stream per-statement results. The
        # public /verify endpoint answers with a single body regardless.
        streaming=True,
        skill=_skill(
            id="verify_statements",
            name="Verify statements",
            description=(
                "Takes a set of statements with their attributes and optional labels. "
                "Returns a True/False verdict, the contributing predictors and their "
                "status, whether reconciliation ran, and an explanation for each "
                "statement, plus aggregate metrics when labels are given."),
            tags=["truthfulness", "classification", "orchestration", "reconciliation"],
            examples=[_EXAMPLE]))


def zero_shot_card(base_url: str) -> AgentCard:
    return _card(
        name="truthclf-zero-shot-predictor",
        description=(
            "Predicts True or False for a batch of statements using a zero-shot LLM "
            "classifier, with a calibrator and decision threshold fitted on a held-out "
            "validation split. Returns a calibrated probability per statement. Scores "
            "any statement; requires no prior knowledge of it."),
        base_url=base_url,
        # Ten statements is a single wave of concurrent provider calls. There
        # are no useful intermediate results: probabilities are consumed as a
        # complete set, so a stream would add framing and deliver one event.
        streaming=False,
        skill=_skill(
            id="predict_zero_shot",
            name="Zero-shot truthfulness prediction",
            description=(
                "Scores a batch of statements with the zero-shot predictor and returns "
                "a calibrated probability and a True/False prediction for each."),
            tags=["truthfulness", "prediction", "zero-shot", "calibrated"],
            examples=[_EXAMPLE]))


def fine_tuned_card(base_url: str) -> AgentCard:
    return _card(
        name="truthclf-fine-tuned-predictor",
        description=(
            "Predicts True or False using the LoRA fine-tuned model, with its own "
            "fitted calibrator. Answers from the probabilities recorded during the "
            "fine-tuned evaluation, which cover the validation and test splits "
            "(3,908 statements). A statement outside that set is reported as "
            "unavailable rather than guessed, so callers should expect partial "
            "coverage on statements of their own."),
        base_url=base_url,
        # A lookup in a loaded dictionary plus a two-parameter calibration map.
        streaming=False,
        skill=_skill(
            id="predict_fine_tuned",
            name="Fine-tuned truthfulness prediction",
            description=(
                "Scores a batch of statements with the fine-tuned predictor. Returns a "
                "calibrated probability per statement, or an explicit unavailable "
                "status for statements it holds no recorded probability for."),
            tags=["truthfulness", "prediction", "fine-tuned", "calibrated",
                  "partial-coverage"],
            examples=[_EXAMPLE]))


def explainer_card(base_url: str) -> AgentCard:
    return _card(
        name="truthclf-explainer",
        description=(
            "Explains predictions by leave-one-field-out occlusion over statement "
            "metadata: each field is removed in turn and the shift in predicted "
            "probability is measured, identifying which input drove the prediction. "
            "Adds the model's own rationale and a cross-check between the two. "
            "Explains the zero-shot predictor by default. Does not vote on verdicts."),
        base_url=base_url,
        # The long pole by an order of magnitude -- six occlusion calls plus a
        # rationale call per statement -- and the only agent whose work
        # decomposes into natural increments, one explanation per statement.
        streaming=True,
        skill=_skill(
            id="explain_predictions",
            name="Explain truthfulness predictions",
            description=(
                "Returns, for each statement, the prediction, the field that drove it, "
                "the per-field probability shifts, and the model's rationale with a "
                "cross-check against the measured driver."),
            tags=["explainability", "occlusion", "attribution", "faithfulness"],
            examples=[_EXAMPLE]))


def card_route(build):
    """A GET handler serving the card, with the request in hand.

    The SDK's own create_agent_card_routes takes a card_modifier, but that hook
    is passed only the card -- it never sees the request, so it cannot derive
    the address from the host that was actually reached. Serving the route here
    keeps that ability; the card is still serialised with the SDK's own
    agent_card_to_dict, so the wire format is unchanged.
    """

    async def handler(request: Request) -> JSONResponse:
        return JSONResponse(agent_card_to_dict(build(resolve_base_url(request))))

    return handler


__all__ = ["AGENT_CARD_WELL_KNOWN_PATH", "card_route", "orchestrator_card",
           "zero_shot_card", "fine_tuned_card", "explainer_card"]

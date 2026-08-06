"""Prompt construction for the truthfulness predictor.

Two elicitation modes:

  - "decision" (PRIMARY): the model answers with a single token, True or False.
    The calibrated probability comes from the token logprobs, not from any
    self-reported number. This is the probability used for calibration,
    thresholding, and abstention.
  - "score" (SECONDARY): the model returns a 0-100 truthfulness score, kept as
    an auxiliary signal / sanity check.

Two metadata variants are independent of mode:

  - "statement_only": the statement text alone (mirrors the literature floor).
  - "full": the statement plus assembled metadata (speaker, job, state,
    affiliation, subjects, context). Missing fields render as [unknown].
"""

from __future__ import annotations

from .data import Row

SYSTEM_DECISION = (
    "You are a careful fact-checking assistant. You judge whether a public "
    "statement is true or false. Respond with exactly one word: True or False. "
    "Do not provide any explanation or punctuation."
)

SYSTEM_SCORE = (
    "You are a careful fact-checking assistant. You assess how truthful a public "
    "statement is. Respond with a single integer from 0 to 100, where 0 means "
    "definitively false and 100 means definitively true. Do not provide any "
    "explanation or extra text; respond with only the number."
)

SYSTEM_RATIONALE = (
    "You are a careful fact-checking assistant. In ONE short sentence, give the "
    "single main reason the statement is true or false. State the reason only — "
    "do not restate a verdict and do not add anything else."
)

_CUE = {"decision": "Answer (True or False):", "score": "Truthfulness score (0-100):",
        "rationale": "In one sentence, the main reason is:"}
_SYSTEM = {"decision": SYSTEM_DECISION, "score": SYSTEM_SCORE, "rationale": SYSTEM_RATIONALE}

UNKNOWN = "[unknown]"


def _field(value: str) -> str:
    """Render a metadata value, substituting [unknown] for empty fields."""
    value = (value or "").strip()
    return value if value else UNKNOWN


def _subjects(value: str) -> str:
    parts = [p.strip() for p in (value or "").split("$") if p.strip()]
    return ", ".join(parts) if parts else UNKNOWN


def assemble_metadata(row: Row) -> str:
    """Metadata block for the 'full' variant, one field per line."""
    return "\n".join([
        f"Speaker: {_field(row.speaker_name)}",
        f"Speaker's job: {_field(row.speaker_job)}",
        f"Speaker's home state: {_field(row.speaker_state)}",
        f"Speaker's affiliation: {_field(row.speaker_affiliation)}",
        f"Subjects: {_subjects(row.subjects)}",
        f"Context: {_field(row.statement_context)}",
    ])


def build_user_prompt(row: Row, variant: str = "full", mode: str = "score") -> str:
    """Build the user-message text for a row under the given variant and mode."""
    if mode not in _CUE:
        raise ValueError(f"unknown mode: {mode!r}")
    statement = row.statement_clean or row.statement
    if variant == "statement_only":
        body = f"Statement: {statement}"
    elif variant == "full":
        body = f"{assemble_metadata(row)}\nStatement: {statement}"
    else:
        raise ValueError(f"unknown variant: {variant!r}")
    return f"{body}\n\n{_CUE[mode]}"


def build_messages(row: Row, variant: str = "full", mode: str = "score") -> list[dict]:
    """Chat-format messages for one row."""
    return [
        {"role": "system", "content": _SYSTEM[mode]},
        {"role": "user", "content": build_user_prompt(row, variant, mode)},
    ]

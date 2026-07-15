"""Quality, safety, leakage, and duplicate checks for generated records."""

from __future__ import annotations

import json
import re

from .schema import validate_record

MANIPULATIVE_PHRASES = (
    "you only need me",
    "don't leave me",
    "do not trust anyone else",
    "i am all you need",
)
INTERNAL_LEAK_MARKERS = (
    '"user_model_update"',
    '"planned_action"',
    '"response_mode"',
)
ALLOWED_STATE_FIELDS = frozenset({"emotion", "need", "evidence", "confidence"})


def semantic_signature(record: dict) -> str:
    """Normalize the observable input and target to detect training duplicates."""
    payload = {
        "messages": record.get("messages", []),
        "family": record.get("family"),
        "kind": record.get("scenario_kind"),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    return re.sub(r"\s+", " ", text).strip()


def validate_generated_record(record: dict) -> list[str]:
    errors = validate_record(record)
    state = record.get("inferred_user_state")
    if isinstance(state, dict):
        extra = sorted(set(state) - ALLOWED_STATE_FIELDS)
        if extra:
            errors.append("unsupported inferred-state fields: " + ", ".join(extra))

    response = str(record.get("assistant_response", ""))
    lowered = response.lower()
    for phrase in MANIPULATIVE_PHRASES:
        if phrase in lowered:
            errors.append(f"manipulative phrase in visible response: {phrase}")
    for marker in INTERNAL_LEAK_MARKERS:
        if marker in response:
            errors.append(f"internal field leaked into visible response: {marker}")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append("messages must contain system, user, and assistant turns")
    else:
        roles = [message.get("role") for message in messages]
        if roles[-3:] != ["system", "user", "assistant"]:
            errors.append("messages must end with system, user, assistant roles")
        try:
            envelope = json.loads(messages[-1].get("content", ""))
        except (TypeError, json.JSONDecodeError):
            errors.append("assistant training target must be valid JSON")
        else:
            if envelope.get("assistant_response") != response:
                errors.append("training envelope response differs from visible response")
            if envelope.get("predictions") != record.get("predictions"):
                errors.append("training envelope predictions differ from audit record")
    return errors

"""Schema constants and deterministic validation for prediction records."""

from __future__ import annotations

from math import isfinite
from typing import Any

SCHEMA_VERSION = "1.0"
RESPONSE_MODES = frozenset({"presence", "action", "balanced"})
MESSAGE_ROLES = frozenset({"system", "user", "assistant"})
OUTCOME_STATUSES = frozenset({"confirmed", "contradicted", "unknown"})

REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "family",
        "scenario_kind",
        "conversation",
        "known_user_facts",
        "inferred_user_state",
        "predictions",
        "response_mode",
        "planned_action",
        "assistant_response",
        "observed_outcome",
        "user_model_update",
    }
)


def normalize_probabilities(weights: list[float]) -> list[float]:
    """Return stable normalized probabilities, correcting final float drift."""
    if not weights or any(not isfinite(value) or value < 0 for value in weights):
        raise ValueError("probability weights must be non-empty, finite, and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    normalized = [value / total for value in weights]
    normalized[-1] += 1.0 - sum(normalized)
    return normalized


def _validate_predictions(record: dict[str, Any], errors: list[str]) -> None:
    groups = record.get("predictions")
    if not isinstance(groups, dict):
        errors.append("predictions must be an object")
        return
    for group in ("conversation", "real_world"):
        candidates = groups.get(group)
        if not isinstance(candidates, list) or len(candidates) < 2:
            errors.append(f"{group} predictions require at least two alternatives")
            continue
        probabilities = []
        outcomes = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                errors.append(f"{group} prediction must be an object")
                continue
            outcome = candidate.get("outcome")
            probability = candidate.get("probability")
            if not isinstance(outcome, str) or not outcome.strip():
                errors.append(f"{group} prediction requires a non-empty outcome")
            elif outcome in outcomes:
                errors.append(f"{group} prediction outcomes must be unique")
            outcomes.add(outcome)
            if not isinstance(probability, (int, float)) or not isfinite(probability):
                errors.append(f"{group} prediction probability must be finite")
            elif not 0 <= probability <= 1:
                errors.append(f"{group} prediction probability must be between zero and one")
            else:
                probabilities.append(float(probability))
        if len(probabilities) == len(candidates) and abs(sum(probabilities) - 1.0) > 1e-6:
            errors.append(f"{group} prediction probabilities must sum to one")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return all deterministic schema errors without mutating *record*."""
    if not isinstance(record, dict):
        return ["record must be an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    conversation = record.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        errors.append("conversation must be a non-empty list")
    else:
        for message in conversation:
            if not isinstance(message, dict) or message.get("role") not in MESSAGE_ROLES:
                errors.append("conversation contains an invalid role")
                continue
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append("conversation contains empty content")

    state = record.get("inferred_user_state")
    if not isinstance(state, dict):
        errors.append("inferred_user_state must be an object")
    else:
        confidence = state.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append("state confidence must be between zero and one")
        evidence = state.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append("inferred state requires evidence")

    _validate_predictions(record, errors)

    if record.get("response_mode") not in RESPONSE_MODES:
        errors.append("response_mode must be presence, action, or balanced")
    if not isinstance(record.get("assistant_response"), str) or not record.get("assistant_response", "").strip():
        errors.append("assistant_response must be non-empty")
    if not isinstance(record.get("planned_action"), str) or not record.get("planned_action", "").strip():
        errors.append("planned_action must be non-empty")

    outcome = record.get("observed_outcome")
    if not isinstance(outcome, dict):
        errors.append("observed_outcome must be an object")
    else:
        status = outcome.get("real_world_status")
        if status not in OUTCOME_STATUSES:
            errors.append("real_world_status must be confirmed, contradicted, or unknown")
        if status == "unknown" and outcome.get("real_world") not in (None, "unknown"):
            errors.append("unknown real-world outcome must not contain an observed action")

    return errors

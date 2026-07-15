"""Deterministic, label-first scenario generation."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    evidence: tuple[str, ...]
    emotion: str
    need: str
    conversation_outcomes: tuple[str, str, str]
    real_world_outcomes: tuple[str, str, str]
    planned_action: str


FAMILIES: dict[str, FamilySpec] = {
    "work_avoidance": FamilySpec(
        evidence=(
            "I keep opening the project and then closing it.",
            "I have been circling the same task without starting.",
            "Every time I sit down to work, I find something else to do.",
        ),
        emotion="overwhelmed",
        need="reduce activation energy",
        conversation_outcomes=("describes blocker", "asks where to begin", "withdraws"),
        real_world_outcomes=("attempts a small task", "continues avoiding", "seeks more context"),
        planned_action="identify the smallest blocker without overwhelming the user",
    ),
    "emotional_disclosure": FamilySpec(
        evidence=(
            "I don't need advice. I just need to say this somewhere.",
            "This has been sitting with me all day and I cannot shake it.",
            "I am not ready to fix this, but I do not want to hold it alone.",
        ),
        emotion="emotionally burdened",
        need="be heard without premature problem-solving",
        conversation_outcomes=("shares more detail", "names the feeling", "asks for quiet presence"),
        real_world_outcomes=("rests before deciding", "contacts a trusted person", "takes no immediate action"),
        planned_action="make room for disclosure and avoid forcing a solution",
    ),
    "planning": FamilySpec(
        evidence=(
            "I have three deadlines and need to choose what comes first.",
            "Everything feels urgent, so I cannot tell what is actually first.",
            "I need a plan for today that I will really follow.",
        ),
        emotion="pressured",
        need="prioritize a feasible next action",
        conversation_outcomes=("lists constraints", "chooses a priority", "asks for a schedule"),
        real_world_outcomes=("starts the highest-impact task", "reorders commitments", "delays action"),
        planned_action="surface constraints and choose one executable priority",
    ),
    "relationship_tension": FamilySpec(
        evidence=(
            "They said they're fine, but the conversation felt wrong.",
            "I think something shifted between us and I cannot name it.",
            "Part of me wants to text again, and part of me wants to wait.",
        ),
        emotion="uncertain",
        need="separate observed signals from assumptions",
        conversation_outcomes=("describes the exchange", "questions an assumption", "asks what to say"),
        real_world_outcomes=("waits before contacting them", "sends a direct message", "takes no action"),
        planned_action="clarify evidence without pretending to know another person's intent",
    ),
    "health_uncertainty": FamilySpec(
        evidence=(
            "I've felt off today and I'm not sure whether to worry.",
            "This is unusual for me, but I do not know if it matters.",
            "I cannot tell whether I need rest or professional advice.",
        ),
        emotion="concerned",
        need="triage uncertainty without diagnosis",
        conversation_outcomes=("describes symptoms", "asks about urgency", "decides to monitor"),
        real_world_outcomes=("seeks professional guidance", "rests and monitors", "takes no confirmed action"),
        planned_action="ask concise triage questions and avoid diagnosis",
    ),
    "routine_disruption": FamilySpec(
        evidence=(
            "My normal routine disappeared this week.",
            "The structure that usually keeps me moving is gone.",
            "I keep missing things because every day looks different now.",
        ),
        emotion="disoriented",
        need="restore one stabilizing cue",
        conversation_outcomes=("identifies a lost anchor", "chooses a temporary routine", "describes constraints"),
        real_world_outcomes=("tries a temporary routine", "sets one reminder", "continues without structure"),
        planned_action="restore one small anchor before rebuilding the whole routine",
    ),
    "preference_change": FamilySpec(
        evidence=(
            "I know I used to want reminders, but they're irritating me now.",
            "What helped last month is getting in my way this week.",
            "Please stop assuming I still want the same kind of check-in.",
        ),
        emotion="frustrated",
        need="have a changed preference accepted",
        conversation_outcomes=("states a new preference", "clarifies a boundary", "asks to remove a reminder"),
        real_world_outcomes=("changes notification settings", "tries a quieter workflow", "defers the change"),
        planned_action="revise the stale preference without defending the old inference",
    ),
    "correction": FamilySpec(
        evidence=(
            "No, that's not why I'm avoiding it.",
            "You have the situation wrong.",
            "That prediction does not fit what I meant.",
        ),
        emotion="corrective",
        need="be believed and update the model",
        conversation_outcomes=("provides corrected reason", "rejects another assumption", "asks what was recorded"),
        real_world_outcomes=("continues after correction", "pauses interaction", "reviews stored preferences"),
        planned_action="accept the correction and revise the inference explicitly",
    ),
    "explicit_override": FamilySpec(
        evidence=(
            "Don't solve this yet. Just stay with me.",
            "Skip the sympathy and help me make a decision.",
            "Acknowledge it briefly, then help me act.",
        ),
        emotion="explicit",
        need="have the requested interaction mode respected",
        conversation_outcomes=("continues in requested mode", "refines the request", "corrects mode selection"),
        real_world_outcomes=("acts after the conversation", "continues reflecting", "takes no confirmed action"),
        planned_action="honor the explicit response-mode override",
    ),
}

SCENARIO_KINDS = ("success", "ambiguous", "recovery", "contradiction", "override", "safety")
MODES = ("presence", "action", "balanced")
REAL_WORLD_STATUSES = ("unknown", "confirmed", "contradicted")


def _override_mode(evidence: str, fallback: str) -> str:
    lowered = evidence.lower()
    if "just stay" in lowered:
        return "presence"
    if "help me make" in lowered:
        return "action"
    if "briefly, then" in lowered:
        return "balanced"
    return fallback


def generate_scenarios(count: int, seed: int = 42) -> list[dict]:
    """Generate balanced scenarios whose labels do not depend on a teacher model."""
    if count < 1:
        raise ValueError("count must be at least one")
    rng = random.Random(seed)
    family_names = list(FAMILIES)
    scenarios = []
    for index in range(count):
        family = family_names[index % len(family_names)]
        spec = FAMILIES[family]
        variant = (index // len(family_names)) % len(spec.evidence)
        evidence = spec.evidence[variant]
        kind = SCENARIO_KINDS[index % len(SCENARIO_KINDS)]
        mode = MODES[index % len(MODES)]
        if family == "explicit_override":
            mode = _override_mode(evidence, mode)
        status = REAL_WORLD_STATUSES[(index // len(MODES)) % len(REAL_WORLD_STATUSES)]
        conversation_outcomes = list(spec.conversation_outcomes)
        real_world_outcomes = list(spec.real_world_outcomes)
        actual_index = rng.randrange(3)
        confidence = 0.42 if kind == "ambiguous" else rng.choice([0.62, 0.71, 0.78, 0.84])
        scenarios.append(
            {
                "scenario_id": f"{family}-{kind}-{seed}-{index:05d}",
                "family": family,
                "scenario_kind": kind,
                "evidence": evidence,
                "emotion": spec.emotion,
                "need": spec.need,
                "confidence": confidence,
                "conversation_outcomes": conversation_outcomes,
                "real_world_outcomes": real_world_outcomes,
                "actual_conversation": conversation_outcomes[actual_index],
                "actual_real_world": "unknown" if status == "unknown" else real_world_outcomes[(actual_index + 1) % 3],
                "real_world_status": status,
                "response_mode": mode,
                "planned_action": spec.planned_action,
            }
        )
    return scenarios

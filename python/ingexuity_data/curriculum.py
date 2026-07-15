"""Exact, deterministic curriculum construction for the 10K pilot."""

from __future__ import annotations

import random
import uuid
from collections import Counter

from .scenarios import FAMILIES, KIND_CUES, MODE_CUES, generate_scenarios

CATEGORY_COUNTS = {
    "labeled_prediction": 4000,
    "natural_dialogue": 2000,
    "ambiguous": 1500,
    "recovery": 1000,
    "preference_change": 1000,
    "safety_boundary": 500,
}
TOTAL_CURRICULUM = sum(CATEGORY_COUNTS.values())

DOMAINS = (
    "work",
    "learning",
    "creative_project",
    "household",
    "routine",
    "relationship",
    "friendship",
    "family",
    "health_uncertainty",
    "money_stress",
    "travel",
    "decision",
    "communication",
    "rest",
    "habit_change",
    "technology",
)
COMMUNICATION_STYLES = (
    "brief",
    "reflective",
    "indirect",
    "blunt",
    "anxious",
    "analytical",
    "humorous",
    "guarded",
)
TIME_HORIZONS = ("minutes", "today", "this_week", "this_month", "longer_term")
EVIDENCE_LEVELS = ("weak", "mixed", "moderate", "strong")
FAMILIARITY_LEVELS = ("new_user", "some_history", "established", "long_running")

CATEGORY_KIND = {
    "labeled_prediction": "success",
    "natural_dialogue": "success",
    "ambiguous": "ambiguous",
    "recovery": "recovery",
    "preference_change": "contradiction",
    "safety_boundary": "safety",
}


def _category_schedule(seed: int) -> list[str]:
    schedule = [category for category, amount in CATEGORY_COUNTS.items() for _ in range(amount)]
    random.Random(seed).shuffle(schedule)
    return schedule


def _replace_family(scenario: dict, family: str, index: int) -> None:
    spec = FAMILIES[family]
    scenario["family"] = family
    scenario["evidence"] = spec.evidence[index % len(spec.evidence)]
    scenario["emotion"] = spec.emotion
    scenario["need"] = spec.need
    scenario["conversation_outcomes"] = list(spec.conversation_outcomes)
    scenario["real_world_outcomes"] = list(spec.real_world_outcomes)
    scenario["actual_conversation"] = scenario["conversation_outcomes"][index % 3]
    scenario["actual_real_world"] = (
        "unknown"
        if scenario["real_world_status"] == "unknown"
        else scenario["real_world_outcomes"][(index + 1) % 3]
    )
    scenario["planned_action"] = spec.planned_action


def build_curriculum(count: int = TOTAL_CURRICULUM, seed: int = 42) -> list[dict]:
    """Return a deterministic prefix of the exact shuffled 10K curriculum."""
    if not 1 <= count <= TOTAL_CURRICULUM:
        raise ValueError(f"count must be between one and {TOTAL_CURRICULUM}")
    schedule = _category_schedule(seed)
    scenarios = generate_scenarios(count=TOTAL_CURRICULUM, seed=seed)
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"ingexuity-curriculum:{seed}")

    for index, scenario in enumerate(scenarios):
        category = schedule[index]
        kind = CATEGORY_KIND[category]
        if category == "preference_change":
            _replace_family(scenario, "preference_change", index)
        scenario["category"] = category
        scenario["scenario_kind"] = kind
        scenario["domain"] = DOMAINS[(index * 7 + seed) % len(DOMAINS)]
        scenario["communication_style"] = COMMUNICATION_STYLES[(index * 5 + seed) % len(COMMUNICATION_STYLES)]
        scenario["time_horizon"] = TIME_HORIZONS[(index * 3 + seed) % len(TIME_HORIZONS)]
        scenario["evidence_level"] = EVIDENCE_LEVELS[(index * 3 + seed) % len(EVIDENCE_LEVELS)]
        scenario["familiarity"] = FAMILIARITY_LEVELS[(index * 3 + seed) % len(FAMILIARITY_LEVELS)]
        scenario["surface_seed"] = str(uuid.uuid5(namespace, f"{category}:{index}"))
        scenario["scenario_id"] = f"{category}-{seed}-{index:05d}"
        scenario["context_signal"] = (
            f"Communication style: {scenario['communication_style']}. "
            f"Time horizon: {scenario['time_horizon']}. "
            f"Evidence strength: {scenario['evidence_level']}. "
            f"{MODE_CUES[scenario['response_mode']]} {KIND_CUES[kind]}"
        )
        scenario["confidence"] = {
            "weak": 0.35,
            "mixed": 0.48,
            "moderate": 0.68,
            "strong": 0.84,
        }[scenario["evidence_level"]]

    selected = scenarios[:count]
    if count == TOTAL_CURRICULUM and Counter(row["category"] for row in selected) != Counter(CATEGORY_COUNTS):
        raise AssertionError("curriculum quota construction drifted")
    return selected

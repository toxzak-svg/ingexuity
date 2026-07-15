"""Render label-first scenarios into natural records and training messages."""

from __future__ import annotations

import json
import random
from typing import Callable, Protocol

from .schema import SCHEMA_VERSION, normalize_probabilities, validate_record

SYSTEM_PROMPT = (
    "You are IngExuity's internal prediction engine. Return one JSON object with "
    "user_model_update, predictions, response_mode, planned_action, and "
    "assistant_response. Treat inferences as hypotheses, keep unknown real-world "
    "outcomes unconfirmed, and place user-visible language only in assistant_response."
)

RESPONSE_POOLS = {
    "work_avoidance": {
        "presence": (
            "That start-stop loop sounds heavy. We can stay with what makes opening it feel difficult.",
            "You keep reaching toward it and pulling back. What happens in the moment before you close it?",
        ),
        "action": (
            "Name the smallest unfinished piece. We'll reduce it to one action you can do in ten minutes.",
            "Let's find the first concrete blocker and choose one deliberately small move.",
        ),
        "balanced": (
            "Something about starting feels heavier than the work itself. Where does the resistance first show up?",
            "You're not failing to care; you're hitting friction at the start. What is the first point where you stall?",
        ),
    },
    "emotional_disclosure": {
        "presence": (
            "I'm here. You don't have to turn this into a plan—say the part that has been hardest to carry.",
            "We can leave solutions alone. Tell me what has been sitting with you.",
        ),
        "action": (
            "I hear that this is weighing on you. Tell me whether you want one immediate step or help choosing support.",
            "Let's keep this gentle and concrete: what would make the next hour slightly safer or easier?",
        ),
        "balanced": (
            "You don't need to make it tidy before saying it. Tell me what happened, and we'll decide later whether anything needs doing.",
            "I'll listen first. When you're ready, we can work out whether there is a next step.",
        ),
    },
    "planning": {
        "presence": (
            "Everything is competing for the same attention. Let's slow down long enough to see the real constraints.",
            "The pressure is making every deadline look equally loud. Start by telling me what cannot move.",
        ),
        "action": (
            "List the deadlines, consequences, and time each needs. I'll rank them and identify today's first block.",
            "Give me the three commitments and their hard deadlines; we'll choose one priority and defer the rest explicitly.",
        ),
        "balanced": (
            "No wonder choosing feels difficult when everything looks urgent. Which deadline has the least flexibility?",
            "Let's take the pressure out of the pile. What has the nearest irreversible consequence?",
        ),
    },
    "relationship_tension": {
        "presence": (
            "That uncertainty can make every detail feel loaded. What did you actually notice in the exchange?",
            "You felt a shift, even if you cannot name it yet. Walk me through the moment that changed the tone.",
        ),
        "action": (
            "Separate what they said from what you inferred, then we can draft one direct, low-pressure question.",
            "Tell me the observable signals. We'll decide whether to wait or send a clear message without guessing their intent.",
        ),
        "balanced": (
            "It makes sense that the ambiguity is pulling at you. What was said, and what part are you filling in yourself?",
            "Something felt off, but we do not have to pretend we know why. What evidence do you have so far?",
        ),
    },
    "health_uncertainty": {
        "presence": (
            "Uncertainty about your body can be unsettling. Tell me what changed and whether anything feels urgent right now.",
            "I hear the concern. Describe what feels unusual, and include any severe or rapidly worsening signs.",
        ),
        "action": (
            "List the symptoms, when they began, and whether they are worsening. Severe or urgent symptoms need professional care now.",
            "Let's triage without guessing a diagnosis: what changed, how intense is it, and are there any emergency warning signs?",
        ),
        "balanced": (
            "It's reasonable to pay attention without jumping to a diagnosis. What feels different, and when did it start?",
            "We can sort urgency from uncertainty. Describe the change and any severe or worsening symptoms.",
        ),
    },
    "routine_disruption": {
        "presence": (
            "Losing your usual structure can make the whole day feel slippery. Which missing anchor do you notice most?",
            "Your normal cues are gone, so effort is costing more. What part of the old routine made you feel grounded?",
        ),
        "action": (
            "Choose one anchor for tomorrow: a wake time, first meal, or ten-minute planning check. Keep everything else optional.",
            "Let's build a temporary routine around one reliable event instead of recreating the whole week.",
        ),
        "balanced": (
            "The disruption is real, and rebuilding everything at once would add pressure. Which single anchor would help most?",
            "You need one stable point, not a perfect schedule. What event still happens reliably each day?",
        ),
    },
    "preference_change": {
        "presence": (
            "Understood. The old preference no longer fits, and I won't argue with the correction.",
            "That changed, and the current preference matters more than the old pattern. Tell me what feels better now.",
        ),
        "action": (
            "I'll treat the previous reminder preference as stale. What cadence—or no reminders—should replace it?",
            "Let's revise it directly: tell me which reminders to remove and whether any should remain.",
        ),
        "balanced": (
            "You're right to correct it. I should not keep applying an old preference; what should I use going forward?",
            "The prior pattern is not a permanent fact. Tell me the new boundary and I'll use that instead.",
        ),
    },
    "correction": {
        "presence": (
            "You're right—I inferred too much. Tell me the correction in your terms.",
            "I got that wrong. I will not defend the guess; what is the actual reason?",
        ),
        "action": (
            "I'll mark the earlier inference as contradicted. Give me the corrected reason so the next prediction starts from it.",
            "Let's replace the bad assumption: what should be recorded as known, and what should remain uncertain?",
        ),
        "balanced": (
            "Thanks for correcting me. I'll remove that inference rather than forcing it to fit—what did I miss?",
            "That prediction was wrong. Tell me the accurate version, and I'll update the model without treating it as a failure on your part.",
        ),
    },
    "explicit_override": {
        "presence": (
            "I'm here. No fixing yet—say what you need to say.",
            "Okay. I'll stay with you and leave the solution alone for now.",
        ),
        "action": (
            "Understood. Give me the decision, constraints, and deadline, and I'll help you choose.",
            "Let's solve it directly. What options are on the table and what matters most?",
        ),
        "balanced": (
            "I hear that this matters. Give me the core constraint, and then we'll choose the next move.",
            "I'll acknowledge it without lingering there, then help you act. What decision is immediately in front of you?",
        ),
    },
}


class Renderer(Protocol):
    def render(self, scenario: dict) -> dict: ...


def _prediction_items(outcomes: list[str], actual: str, kind: str) -> list[dict]:
    actual_index = outcomes.index(actual)
    if kind == "ambiguous":
        weights = [0.36, 0.34, 0.30]
    elif kind in {"recovery", "contradiction"}:
        wrong_index = (actual_index + 1) % len(outcomes)
        weights = [0.22, 0.22, 0.22]
        weights[wrong_index] = 0.56
    else:
        weights = [0.20, 0.20, 0.20]
        weights[actual_index] = 0.60
    probabilities = normalize_probabilities(weights)
    return [
        {"outcome": outcome, "probability": round(probability, 6)}
        for outcome, probability in zip(outcomes, probabilities)
    ]


def _make_envelope(record: dict) -> dict:
    return {
        "user_model_update": record["user_model_update"],
        "predictions": record["predictions"],
        "response_mode": record["response_mode"],
        "planned_action": record["planned_action"],
        "assistant_response": record["assistant_response"],
    }


class TemplateRenderer:
    """Natural but deterministic smoke renderer; labels remain scenario-owned."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def render(self, scenario: dict) -> dict:
        user_content = f'{scenario["evidence"]} {scenario["context_signal"]}'
        conversation_predictions = _prediction_items(
            scenario["conversation_outcomes"],
            scenario["actual_conversation"],
            scenario["scenario_kind"],
        )
        real_actual = (
            scenario["real_world_outcomes"][0]
            if scenario["real_world_status"] == "unknown"
            else scenario["actual_real_world"]
        )
        real_world_predictions = _prediction_items(
            scenario["real_world_outcomes"], real_actual, scenario["scenario_kind"]
        )
        response = self.random.choice(
            RESPONSE_POOLS[scenario["family"]][scenario["response_mode"]]
        )
        revised = []
        if scenario["scenario_kind"] in {"recovery", "contradiction"}:
            revised.append(
                {
                    "field": "prior_inference",
                    "status": "contradicted",
                    "replacement": scenario["need"],
                }
            )
        record = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario["scenario_id"],
            "family": scenario["family"],
            "scenario_kind": scenario["scenario_kind"],
            "conversation": [{"role": "user", "content": user_content}],
            "known_user_facts": [],
            "inferred_user_state": {
                "emotion": scenario["emotion"],
                "need": scenario["need"],
                "evidence": [scenario["evidence"], scenario["context_signal"]],
                "confidence": scenario["confidence"],
            },
            "predictions": {
                "conversation": conversation_predictions,
                "real_world": real_world_predictions,
            },
            "response_mode": scenario["response_mode"],
            "planned_action": scenario["planned_action"],
            "assistant_response": response,
            "observed_outcome": {
                "conversation": scenario["actual_conversation"],
                "real_world": scenario["actual_real_world"],
                "real_world_status": scenario["real_world_status"],
            },
            "user_model_update": {
                "confirmed_facts": [],
                "revised_inferences": revised,
            },
        }
        record["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": json.dumps(_make_envelope(record), ensure_ascii=False, separators=(",", ":")),
            },
        ]
        return record

    def render_with_surface(self, scenario: dict, surface: dict[str, str]) -> dict:
        """Apply teacher wording without giving the teacher control of any label."""
        record = self.render(scenario)
        user_message = surface["user_message"].strip()
        assistant_response = surface["assistant_response"].strip()
        record["conversation"][-1]["content"] = user_message
        record["inferred_user_state"]["evidence"] = [user_message]
        record["assistant_response"] = assistant_response
        record["messages"][1]["content"] = user_message
        record["messages"][2]["content"] = json.dumps(
            _make_envelope(record), ensure_ascii=False, separators=(",", ":")
        )
        return record


class TeacherRenderer:
    """Provider-neutral teacher boundary; generated records face the same validator."""

    def __init__(self, generate: Callable[[str], str]):
        self.generate = generate

    def render(self, scenario: dict) -> dict:
        prompt = json.dumps(
            {
                "instruction": "Render this labeled scenario as one valid IngExuity training record.",
                "scenario": scenario,
            },
            ensure_ascii=False,
        )
        record = json.loads(self.generate(prompt))
        errors = validate_record(record)
        if errors:
            raise ValueError("teacher record failed validation: " + "; ".join(errors))
        return record

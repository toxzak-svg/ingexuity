import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingexuity_data.schema import normalize_probabilities, validate_record
from ingexuity_data.scenarios import generate_scenarios
from ingexuity_data.render import TemplateRenderer


def valid_record():
    return {
        "schema_version": "1.0",
        "scenario_id": "work-0001",
        "family": "work_avoidance",
        "scenario_kind": "success",
        "conversation": [
            {"role": "user", "content": "I keep opening the project and closing it."}
        ],
        "known_user_facts": [],
        "inferred_user_state": {
            "emotion": "overwhelmed",
            "need": "reduce activation energy",
            "evidence": ["repeatedly opens and closes project"],
            "confidence": 0.76,
        },
        "predictions": {
            "conversation": [
                {"outcome": "describes blocker", "probability": 0.5},
                {"outcome": "asks where to begin", "probability": 0.3},
                {"outcome": "withdraws", "probability": 0.2},
            ],
            "real_world": [
                {"outcome": "attempts small task", "probability": 0.4},
                {"outcome": "continues avoiding", "probability": 0.35},
                {"outcome": "seeks context", "probability": 0.25},
            ],
        },
        "response_mode": "balanced",
        "planned_action": "identify the smallest blocker",
        "assistant_response": "Where does the resistance first show up?",
        "observed_outcome": {
            "conversation": "describes blocker",
            "real_world": "unknown",
            "real_world_status": "unknown",
        },
        "user_model_update": {
            "confirmed_facts": [],
            "revised_inferences": [],
        },
    }


def test_normalize_probabilities_sums_to_one():
    values = normalize_probabilities([4, 3, 3])
    assert values == [0.4, 0.3, 0.3]


def test_rejects_unconfirmed_real_world_outcome():
    record = valid_record()
    record["observed_outcome"]["real_world"] = "started project"
    errors = validate_record(record)
    assert "unknown real-world outcome must not contain an observed action" in errors


def test_valid_record_has_no_errors():
    assert validate_record(valid_record()) == []


def test_scenarios_are_reproducible_and_balanced():
    first = generate_scenarios(count=100, seed=42)
    second = generate_scenarios(count=100, seed=42)
    assert first == second
    assert len({item["scenario_id"] for item in first}) == 100
    assert {item["response_mode"] for item in first} == {
        "presence",
        "action",
        "balanced",
    }
    assert any(item["real_world_status"] == "unknown" for item in first)
    assert any(item["scenario_kind"] == "recovery" for item in first)


def test_every_scenario_has_competing_predictions():
    for scenario in generate_scenarios(count=20, seed=3):
        assert len(scenario["conversation_outcomes"]) == 3
        assert len(scenario["real_world_outcomes"]) == 3
        assert scenario["actual_conversation"] in scenario["conversation_outcomes"]
        if scenario["real_world_status"] != "unknown":
            assert scenario["actual_real_world"] in scenario["real_world_outcomes"]


def test_renderer_keeps_internal_fields_out_of_visible_response():
    scenario = generate_scenarios(1, seed=7)[0]
    record = TemplateRenderer(seed=7).render(scenario)
    assert record["assistant_response"]
    assert "confidence" not in record["assistant_response"].lower()
    assert "predictions" not in record["assistant_response"].lower()
    assert validate_record(record) == []


def test_renderer_creates_structured_training_target():
    scenario = generate_scenarios(1, seed=9)[0]
    record = TemplateRenderer(seed=9).render(scenario)
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    envelope = json.loads(record["messages"][-1]["content"])
    assert envelope["assistant_response"] == record["assistant_response"]
    assert envelope["response_mode"] == record["response_mode"]
    assert envelope["predictions"] == record["predictions"]

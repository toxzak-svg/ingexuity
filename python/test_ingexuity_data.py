import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingexuity_data.schema import normalize_probabilities, validate_record
from ingexuity_data.scenarios import generate_scenarios
from ingexuity_data.render import TemplateRenderer
from ingexuity_data.build import build_dataset
from ingexuity_data.curriculum import CATEGORY_COUNTS, build_curriculum
from collections import Counter


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
    mode_counts = {
        mode: sum(item["response_mode"] == mode for item in first)
        for mode in {"presence", "action", "balanced"}
    }
    assert max(mode_counts.values()) - min(mode_counts.values()) <= 2


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


def test_build_writes_family_isolated_splits_and_manifest(tmp_path):
    result = build_dataset(output_dir=tmp_path, count=100, seed=42)
    train_families = set(result["split_families"]["train"])
    eval_families = set(result["split_families"]["eval"])
    test_families = set(result["split_families"]["test"])
    assert train_families.isdisjoint(eval_families | test_families)
    assert eval_families.isdisjoint(test_families)
    assert result["accepted"] == 100
    assert result["rejected"] == 0
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "train.jsonl").is_file()


def test_built_rows_are_model_ready_messages(tmp_path):
    build_dataset(output_dir=tmp_path, count=30, seed=5)
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(json.loads(row["messages"][-1]["content"])["predictions"] for row in rows)


def test_10k_curriculum_has_exact_approved_quotas():
    scenarios = build_curriculum(count=10_000, seed=42)
    assert Counter(row["category"] for row in scenarios) == CATEGORY_COUNTS
    assert len({row["scenario_id"] for row in scenarios}) == 10_000


def test_curriculum_has_observable_diversity_without_counter_language():
    scenarios = build_curriculum(count=10_000, seed=42)
    assert len({row["surface_seed"] for row in scenarios}) == 10_000
    assert len({row["domain"] for row in scenarios}) >= 12
    assert len({row["communication_style"] for row in scenarios}) >= 6
    assert all("noticed this" not in row["context_signal"].lower() for row in scenarios)


def test_curriculum_benchmark_is_deterministic_subset():
    first = build_curriculum(count=200, seed=42)
    second = build_curriculum(count=200, seed=42)
    assert first == second
    assert len(first) == 200

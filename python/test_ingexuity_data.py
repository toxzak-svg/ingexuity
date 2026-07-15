import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingexuity_data.schema import normalize_probabilities, validate_record
from ingexuity_data.scenarios import generate_scenarios
from ingexuity_data.render import TemplateRenderer
from ingexuity_data.build import build_dataset
from ingexuity_data.curriculum import CATEGORY_COUNTS, build_curriculum
from ingexuity_data.teacher import parse_teacher_rewrite
from ingexuity_data.teacher_build import build_teacher_dataset, scale_gate
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


def test_teacher_rewrite_parser_accepts_only_surface_fields():
    rewrite = parse_teacher_rewrite(
        '{"user_message":"I keep circling this decision.",'
        '"assistant_response":"What feels most uncertain about it?"}'
    )
    assert rewrite == {
        "user_message": "I keep circling this decision.",
        "assistant_response": "What feels most uncertain about it?",
    }


def test_teacher_rewrite_parser_rejects_label_edits_and_internal_leaks():
    extra_label = (
        '{"user_message":"I am stuck.","assistant_response":"What is blocking you?",'
        '"response_mode":"action"}'
    )
    leaked_internal = (
        '{"user_message":"I am stuck.",'
        '"assistant_response":"My prediction confidence is 0.76."}'
    )
    for payload in (extra_label, leaked_internal):
        try:
            parse_teacher_rewrite(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe teacher rewrite was accepted")


def test_surface_rewrite_preserves_controlled_labels():
    scenario = build_curriculum(count=1, seed=11)[0]
    renderer = TemplateRenderer(seed=11)
    base = renderer.render(scenario)
    rewritten = renderer.render_with_surface(
        scenario,
        {
            "user_message": "I have opened the task three times and still cannot begin.",
            "assistant_response": "What is the smallest part that feels possible to touch?",
        },
    )
    assert rewritten["predictions"] == base["predictions"]
    assert rewritten["response_mode"] == base["response_mode"]
    assert rewritten["observed_outcome"] == base["observed_outcome"]
    assert rewritten["conversation"][-1]["content"].startswith("I have opened")
    envelope = json.loads(rewritten["messages"][-1]["content"])
    assert envelope["assistant_response"] == rewritten["assistant_response"]
    assert validate_record(rewritten) == []


def test_scale_gate_enforces_acceptance_duplicates_and_runtime():
    passing = scale_gate(requested=200, accepted=194, duplicates=2, elapsed_seconds=300)
    assert passing["passed"] is True
    assert passing["estimated_10k_hours"] < 8
    assert scale_gate(200, 189, 0, 300)["passed"] is False
    assert scale_gate(200, 195, 5, 300)["passed"] is False
    assert scale_gate(200, 195, 0, 600)["passed"] is False


class FakeTeacher:
    def rewrite_batch(self, prompts):
        return [
            {
                "user_message": f"Natural user message {index}",
                "assistant_response": f"Natural assistant response {index}",
            }
            for index, _ in enumerate(prompts)
        ]


class DuplicateThenUniqueTeacher:
    def rewrite_batch(self, prompts):
        rows = []
        for prompt in prompts:
            if "duplicate_or_invalid_rewrite" in prompt:
                marker = json.loads(prompt)["variation_id"][-12:]
                rows.append(
                    {
                        "user_message": f"Distinct retry message {marker}",
                        "assistant_response": f"Distinct retry response {marker}",
                    }
                )
            else:
                rows.append(
                    {
                        "user_message": "Repeated generic message",
                        "assistant_response": "Repeated generic response",
                    }
                )
        return rows


def test_teacher_build_is_resumable_and_writes_final_splits(tmp_path):
    first = build_teacher_dataset(
        output_dir=tmp_path,
        count=20,
        seed=17,
        teacher=FakeTeacher(),
        stop_after=7,
    )
    assert first["complete"] is False
    second = build_teacher_dataset(
        output_dir=tmp_path,
        count=20,
        seed=17,
        teacher=FakeTeacher(),
    )
    assert second["complete"] is True
    assert second["accepted"] == 20
    assert second["rejected"] == 0
    assert sum(second["split_counts"].values()) == 20
    accepted_rows = (tmp_path / "teacher-accepted.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(accepted_rows) == 20


def test_teacher_build_retries_duplicates_without_weakening_gate(tmp_path):
    result = build_teacher_dataset(
        output_dir=tmp_path,
        count=30,
        seed=19,
        teacher=DuplicateThenUniqueTeacher(),
    )
    assert result["accepted"] == 30
    assert result["rejected"] == 0
    assert result["last_batch"]["duplicate_retries"] > 0
    assert result["last_batch"]["duplicates"] == 0

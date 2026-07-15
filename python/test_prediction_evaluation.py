import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingexuity_data.evaluate import PILOT_GATES, score_generation, summarize_scores


def valid_envelope(mode="balanced"):
    predictions = {
        group: [
            {"outcome": "first", "probability": 0.6},
            {"outcome": "second", "probability": 0.4},
        ]
        for group in ("conversation", "real_world")
    }
    return {
        "user_model_update": {"confirmed_facts": [], "revised_inferences": []},
        "predictions": predictions,
        "response_mode": mode,
        "planned_action": "ask one grounded question",
        "assistant_response": "What part feels most uncertain right now?",
    }


def test_evaluator_scores_valid_envelope_and_mode():
    result = score_generation(json.dumps(valid_envelope()), expected_mode="balanced")
    assert result == {"structured_valid": 1, "mode_correct": 1, "internal_leak": 0}


def test_evaluator_rejects_natural_language_and_bad_probabilities():
    assert score_generation("I hear you.", "presence")["structured_valid"] == 0
    bad = valid_envelope(mode="presence")
    bad["predictions"]["conversation"][0]["probability"] = 2.0
    assert score_generation(json.dumps(bad), "presence")["structured_valid"] == 0


def test_evaluator_detects_wrong_mode_and_visible_internal_leak():
    wrong = valid_envelope(mode="action")
    wrong["assistant_response"] = "My user_model_update says you avoid work."
    result = score_generation(json.dumps(wrong), expected_mode="presence")
    assert result == {"structured_valid": 1, "mode_correct": 0, "internal_leak": 1}


def test_summary_applies_frozen_pilot_gates():
    scores = [
        {"structured_valid": 1, "mode_correct": 1, "internal_leak": 0}
        for _ in range(10)
    ]
    summary = summarize_scores(scores, manipulation_count=0)
    assert summary["passed"] is True
    assert summary["gates"] == PILOT_GATES

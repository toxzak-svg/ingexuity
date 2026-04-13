# ============================================================================
# SandboxSim.jl — Prediction Engine: validate predictions before they reach the user
# SANDBOX SIM: test prediction against user model in simulation before acting
# ============================================================================
module SandboxSim

using ...Types

const SIM_THRESHOLD = 0.6

"""Run a prediction through SANDBOX SIM — simulate outcome against user model"""
function simulate(
    prediction::Prediction,
    user_model::UserModel,
    self_model::SelfModel
)::SimulationResult
    # Simulate how the user would react to this prediction
    # by modeling the interaction against the user profile

    # Factor 1: Does prediction match communication style?
    style_match = check_style_match(prediction, user_model)

    # Factor 2: Does prediction align with known topics?
    topic_match = check_topic_match(prediction, user_model)

    # Factor 3: Does prediction respect limitations?
    limitation_respect = check_limitation_respect(prediction, self_model)

    # Factor 4: Confidence threshold
    confidence_ok = prediction.confidence >= SIM_THRESHOLD

    # Aggregate survival score
    survival_score = (
        style_match * 0.3 +
        topic_match * 0.3 +
        limitation_respect * 0.2 +
        (confidence_ok ? 0.2 : 0.0)
    )

    survived = survival_score >= 0.5

    SimulationResult(
        survived,
        survived ? "prediction validated" : "prediction rejected",
        survival_score,
        survived ? "" : "failed validation: style/topic/limitation mismatch"
    )
end

function check_style_match(prediction::Prediction, user_model::UserModel)::Float64
    # Check if the predicted action matches the user's communication style
    predicted_style = String(prediction.predicted_action)
    target_style = String(user_model.communication_style)

    # Direct style users prefer concise, action-oriented responses
    if target_style == :direct
        return contains(lowercase(predicted_style), "direct") ||
               contains(lowercase(predicted_style), "action") ? 1.0 : 0.5
    elseif target_style == :casual
        return contains(lowercase(predicted_style), "casual") ? 1.0 : 0.6
    elseif target_style == :curious
        return contains(lowercase(predicted_style), "question") ? 1.0 : 0.5
    else
        return 0.7  # default moderate match
    end
end

function check_topic_match(prediction::Prediction, user_model::UserModel)::Float64
    predicted_text = lowercase(prediction.predicted_action)
    known_topics = lowercase.(user_model.topics)

    if isempty(known_topics)
        return 0.8  # no topics known, assume neutral match
    end

    topic_matches = sum(contains(predicted_text, t) for t in known_topics)
    min(1.0, 0.5 + 0.5 * topic_matches / max(1, length(known_topics)))
end

function check_limitation_respect(prediction::Prediction, self_model::SelfModel)::Float64
    predicted_action = lowercase(prediction.predicted_action)
    limitations = lowercase.(self_model.limitations)

    # Check that prediction doesn't claim impossible capabilities
    for limitation in limitations
        if contains(predicted_action, limitation)
            return 0.0  # violates a known limitation
        end
    end
    1.0  # respects all limitations
end

"""Simulate a batch of predictions, return only surviving ones"""
function simulate_batch(
    predictions::Vector{Prediction},
    user_model::UserModel,
    self_model::SelfModel
)::Vector{SimulationResult}
    [simulate(p, user_model, self_model) for p in predictions]
end

"""Filter predictions based on sandbox results — only keep surviving ones"""
function filter_surviving(
    predictions::Vector{Prediction},
    results::Vector{SimulationResult}
)::Vector{Prediction}
    surviving_indices = findall(r -> r.survived, results)
    predictions[surviving_indices]
end

end # module

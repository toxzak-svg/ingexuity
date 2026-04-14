# ============================================================================
# UserModel.jl — IngEnuity's model of the human
# ============================================================================
module UserModel

export update, is_stressed

function update(user_model::UserModel, human_input, comprehension)::UserModel
    raw = human_input.raw
    words = split(lowercase(raw))

    # Track topics
    topic = comprehension[:topic]
    if topic != "general" && topic ∉ user_model.topics
        push!(user_model.topics, topic)
    end

    # Track emotional patterns
    emotional = user_model.emotional_patterns

    # Quiet detection
    if length(raw) < 10
        emotional["quiet_count"] = get(emotional, "quiet_count", 0) + 1
    else
        emotional["quiet_count"] = 0
    end
    emotional["is_quiet"] = emotional["quiet_count"] > 3

    # Stress detection
    stress_words = ["stuck", "frustrated", "stressed", "worried", "overwhelmed", "can't", "impossible"]
    if any(w in words for w in stress_words)
        push!(unique!(get!(emotional, "stress_triggers", String[])), "overwhelm_detected")
    end

    user_model.emotional_patterns = emotional
    user_model.prediction_confidence = clamp(user_model.prediction_confidence + 0.01, 0.0, 1.0)

    user_model
end

function is_stressed(user_model::UserModel)::Bool
    emotional = user_model.emotional_patterns
    quiet_ratio = get(emotional, "quiet_count", 0) / max(1, length(user_model.topics))
    stress_triggers = get(emotional, "stress_triggers", String[])
    !isempty(stress_triggers) || quiet_ratio > 0.7
end

end # module

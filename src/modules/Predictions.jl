# ============================================================================
# Predictions.jl — Prediction engine
# ============================================================================
module Predictions

export predict, update_from_outcome!

function predict(user_model, internal_emotional, precog_preds, sandbox_results; context=Dict())
    predictions = Prediction[]

    # User model based predictions
    style_conf = user_model.prediction_confidence
    if style_conf > 0.5
        push!(predictions, Prediction(
            "will respond in direct style",
            "matching communication pattern",
            style_conf,
            [:user_model],
            now()
        ))
    end

    # Topic continuation
    if !isempty(user_model.topics)
        push!(predictions, Prediction(
            "will ask about $(user_model.topics[end])",
            "topic interest",
            0.7,
            [:user_model],
            now()
        ))
    end

    # Emotional state predictions
    if internal_emotional.stress_level > 0.5
        push!(predictions, Prediction(
            "user is stressed — be direct, avoid condescension",
            "emotional support",
            internal_emotional.stress_level,
            [:internal_emotional],
            now()
        ))
    end

    if internal_emotional.arousal > 0.7
        push!(predictions, Prediction(
            "user is highly engaged — respond substantively",
            "high engagement",
            internal_emotional.arousal,
            [:internal_emotional],
            now()
        ))
    end

    predictions
end

function update_from_outcome!(state, prediction, was_correct)
    state.intelligence.total_predictions += 1
    if was_correct
        state.intelligence.correct_predictions += 1
    end
    state.intelligence.accuracy = state.intelligence.correct_predictions / max(1, state.intelligence.total_predictions)
    state.intelligence.last_updated = now()
end

end # module

# ============================================================================
# Predictions.jl — Prediction Engine: converge inputs to generate live predictions
# PRIMARY FUNCTION
# ============================================================================
module Predictions

using ...Types

"""Generate predictions about the user's next state from converged inputs"""
function predict(
    user_model::UserModel,
    internal_emotional::InternalEmotional,
    precognition_preds::Vector{Prediction},
    sandbox_results::Vector{SimulationResult};
    context::Dict{Symbol, Any}=Dict{Symbol, Any}()
)::Vector{Prediction}
    predictions = Prediction[]

    # Source 1: User Model patterns
    append!(predictions, predict_from_user_model(user_model, context))

    # Source 2: Precognition (long-range trajectory)
    append!(predictions, predict_from_precognition(precognition_preds))

    # Source 3: Internal/Emotional state
    append!(predictions, predict_from_emotional(internal_emotional))

    # Source 4: Sandbox validation (filter out failed predictions)
    append!(predictions, predict_from_sandbox(sandbox_results))

    predictions
end

function predict_from_user_model(
    user_model::UserModel;
    context::Dict{Symbol, Any}=Dict{Symbol, Any}()
)::Vector{Prediction}
    predictions = Prediction[]
    raw = get(context, :latest_input, "") |> something("", String)

    # Predict communication style adaptation
    style_confidence = user_model.prediction_confidence
    if style_confidence > 0.6
        push!(predictions, Prediction(
            "will respond in $(user_model.communication_style) style",
            "matching my communication pattern",
            style_confidence,
            [:user_model],
            now()
        ))
    end

    # Predict topic continuation
    if !isempty(user_model.topics)
        top_topic = user_model.topics[end]
        push!(predictions, Prediction(
            "will ask more about $(top_topic)",
            "topic interest continuation",
            0.7,
            [:user_model],
            now()
        ))
    end

    # Predict temporal engagement
    if haskey(user_model.temporal_patterns, "active_hours")
        # simple check — should enhance with proper time handling
        current_hour = Dates.hour(now())
        active_hours = user_model.temporal_patterns["active_hours"]
        if current_hour in active_hours
            push!(predictions, Prediction(
                "is likely in an active engagement period",
                "user available for conversation",
                0.8,
                [:user_model],
                now()
            ))
        end
    end

    predictions
end

function predict_from_precognition(precog_preds::Vector{Prediction})::Vector{Prediction}
    # Filter precog predictions to only high-confidence ones
    [p for p in precog_preds if p.confidence > 0.6]
end

function predict_from_emotional(internal_emotional::InternalEmotional)::Vector{Prediction}
    predictions = Prediction[]

    # Predict engagement level based on arousal
    if internal_emotional.arousal > 0.7
        push!(predictions, Prediction(
            "user is highly engaged — respond substantively",
            "high engagement",
            internal_emotional.arousal,
            [:internal_emotional],
            now()
        ))
    elseif internal_emotional.arousal < 0.3
        push!(predictions, Prediction(
            "user is low engagement — keep response brief",
            "low engagement",
            1.0 - internal_emotional.arousal,
            [:internal_emotional],
            now()
        ))
    end

    # Predict stress response
    if internal_emotional.stress_level > 0.6
        push!(predictions, Prediction(
            "user may be stressed — be direct, avoid condescension",
            "emotional support",
            internal_emotional.stress_level,
            [:internal_emotional],
            now()
        ))
    end

    predictions
end

function predict_from_sandbox(sandbox_results::Vector{SimulationResult})::Vector{Prediction}
    # Only include predictions that survived sandbox simulation
    [p for p in sandbox_results if p.survived]
end

"""Update prediction state based on outcomes (feedback loop)"""
function update_from_outcome!(
    state::PredictionState,
    prediction::Prediction,
    was_correct::Bool
)
    state.intelligence.total_predictions += 1
    if was_correct
        state.intelligence.correct_predictions += 1
    end
    state.intelligence.accuracy = state.intelligence.correct_predictions / state.intelligence.total_predictions
    state.intelligence.last_updated = now()
end

end # module

# ============================================================================
# SandboxSim.jl — Simulate prediction outcomes before responding (Phase 2)
# v2: Actual validation using user model patterns and emotional state
# ============================================================================
module SandboxSim

using ..Types: Prediction, SimulationResult, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType

export simulate_batch, filter_surviving, validate_prediction, validate_batch

function validate_prediction(pred::Prediction, user_model::UserModelType,
                            internal::InternalEmotionalType)::SimulationResult
    feedback = ""
    survival_score = pred.confidence

    if :user_model in pred.source
        emotional = user_model.emotional_patterns
        if pred.predicted_action == "user needs directness, not sympathy"
            if internal.stress_level < 0.4
                survival_score -= 0.3
                feedback *= "prediction doesn't match current stress level; "
            end
        elseif occursin("deflect", lowercase(pred.predicted_action))
            if get(emotional, "deflection_history", 0) == 0
                survival_score -= 0.2
                feedback *= "no deflection history in user model; "
            end
        elseif occursin("withdrawn", lowercase(pred.predicted_action)) || occursin("quiet", lowercase(pred.predicted_action))
            if !get(emotional, "is_quiet", false)
                survival_score -= 0.25
                feedback *= "user not currently quiet; "
            end
        elseif occursin("acknowledgment", lowercase(pred.predicted_action)) || occursin("quiet user", lowercase(pred.predicted_action))
            if get(emotional, "quiet_count", 0) < 2
                survival_score -= 0.2
                feedback *= "user not showing quiet patterns; "
            end
        end
    end

    if :internal_emotional in pred.source
        if occursin("engaged", lowercase(pred.predicted_action)) || occursin("high engagement", lowercase(pred.predicted_need))
            if internal.arousal < 0.5
                survival_score -= 0.3
                feedback *= "user not currently highly engaged; "
            end
        elseif occursin("negative", lowercase(pred.predicted_action)) || occursin("negative headspace", lowercase(pred.predicted_need))
            if internal.valence > -0.2
                survival_score -= 0.25
                feedback *= "user not in negative headspace; "
            end
        elseif occursin("stress", lowercase(pred.predicted_action)) || occursin("stress", lowercase(pred.predicted_need))
            if internal.stress_level < 0.3
                survival_score -= 0.2
                feedback *= "user not currently stressed; "
            end
        end
    end

    if :memory in pred.source || :user_model in pred.source
        if occursin("topic", lowercase(pred.predicted_action))
            if isempty(user_model.topics)
                survival_score -= 0.3
                feedback *= "no topic history to continue; "
            end
        end
    end

    survived = survival_score > 0.35
    outcome = survived ? "positive outcome expected" : "prediction too uncertain for action"
    if !isempty(feedback)
        outcome *= " — $feedback"
    end

    SimulationResult(survived, outcome, survival_score, feedback)
end

function simulate_batch(predictions, user_model::UserModelType,
                        internal::InternalEmotionalType)::Vector{SimulationResult}
    results = SimulationResult[]
    for pred in predictions
        result = validate_prediction(pred, user_model, internal)
        push!(results, result)
    end
    results
end

function simulate_batch(predictions, user_model, self_model)
    results = SimulationResult[]
    for pred in predictions
        survived = pred.confidence > 0.4
        push!(results, SimulationResult(
            survived,
            survived ? "positive outcome expected" : "outcome uncertain",
            pred.confidence,
            survived ? "passed basic validation" : "confidence too low"
        ))
    end
    results
end

function validate_batch(predictions, user_model::UserModelType,
                       internal::InternalEmotionalType)::Vector{SimulationResult}
    simulate_batch(predictions, user_model, internal)
end

function validate_batch(predictions, user_model, self_model)::Vector{SimulationResult}
    simulate_batch(predictions, user_model, self_model)
end

function filter_surviving(predictions, results)
    surviving = Prediction[]
    for (pred, result) in zip(predictions, results)
        if result.survived
            push!(surviving, pred)
        end
    end
    surviving
end

end # module
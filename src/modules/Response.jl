# ============================================================================
# Response.jl — Output: formulate the response, shaped by Understanding
# ============================================================================
module Response

using ...Types

"""Formulate a response given the prediction and Understanding output"""
function formulate(
    predictions::Vector{Prediction},
    understanding::Dict{Symbol, Any};
    tone::ResponseTone=:direct
)::Response
    if isempty(predictions)
        # No prediction — generic response
        return Response(
            "I'm not sure what you need. Could you tell me more?",
            tone,
            0.5,
            0.3,
            true  # should_retry — loop back to Research
        )
    end

    # Get the highest-confidence prediction
    best_pred = argmax(p -> p.confidence, predictions)

    # Build content based on the prediction
    content = build_response_content(best_pred, understanding)

    # Check if this should loop back to Research
    should_retry = best_pred.confidence < 0.7

    Response(
        content,
        tone,
        voice_modulation_from_confidence(best_pred.confidence),
        best_pred.confidence,
        should_retry
    )
end

function build_response_content(
    prediction::Prediction,
    understanding::Dict{Symbol, Any}
)::String
    action = lowercase(prediction.predicted_action)
    need = lowercase(prediction.predicted_need)
    intent_type = get(understanding, :intent, :statement)::Symbol

    if intent_type == :question
        return "You're asking about $need. I understand — let me think about that."
    elseif intent_type == :memory_access
        return "You want me to recall $need. I remember it."
    elseif intent_type == :request
        return "You're looking for $action. I can help with that."
    elseif contains(action, "engaged")
        return "You seem engaged right now — I'll give you a full response."
    elseif contains(action, "brief")
        return "Keeping it short for you."
    else
        return "Based on what I understand: $action."
    end
end

function voice_modulation_from_confidence(confidence::Float64)::Float64
    # Higher confidence → more assertive voice
    # Lower confidence → softer, more questioning voice
    clamp(0.3 + 0.7 * confidence, 0.0, 1.0)
end

"""Adjust response tone based on internal emotional state"""
function adjust_tone(
    response::Response,
    internal_emotional::InternalEmotional
)::Response
    # If user is stressed, be more direct
    if internal_emotional.stress_level > 0.5 && response.tone == :playful
        return Response(
            response.content,
            :direct,
            response.voice_modulation,
            response.confidence,
            response.should_retry
        )
    end

    # If user is positive and engaged, match with warmth
    if internal_emotional.valence > 0.3 && internal_emotional.arousal > 0.6
        return Response(
            response.content,
            :warm,
            response.voice_modulation,
            response.confidence,
            response.should_retry
        )
    end

    response
end

"""Route: should this response loop back to Research for adjustment?"""
function should_retry(response::Response)::Bool
    response.should_retry
end

end # module

# ============================================================================
# Output.jl — Output Layer: final output to the user
# ============================================================================
module Output

using ...Types

"""Render the final output to the user"""
function render(
    response::Response,
    understanding::Dict{Symbol, Any};
    voice_enabled::Bool=true
)::Output
    text = response.content

    # Apply voice modulation if enabled
    modulated_text = voice_enabled ? apply_voice_modulation(text, response) : text

    Output(modulated_text, voice_enabled, now())
end

function apply_voice_modulation(text::String, response::Response)::String
    modulation = response.voice_modulation

    if modulation > 0.8
        # High confidence — full, complete sentences, no hedging
        text
    elseif modulation > 0.5
        # Medium confidence — balanced
        text
    else
        # Low confidence — softer, more qualified
        if !occursin("I think", text) && !occursin("maybe", text)
            "I think " * text
        else
            text
        end
    end
end

"""Build the full Output from a conversation state"""
function build(state::ConversationState)::Output
    # Get predictions
    preds = state.prediction_state.current_predictions

    # Get Understanding
    understanding = Dict{Symbol, Any}(
        :intent => :statement,
        :topic => "general"
    )

    # Formulate response
    resp = Response.formulate(preds, understanding; tone=:direct)

    # Render output
    render(resp, understanding; voice_enabled=true)
end

end # module

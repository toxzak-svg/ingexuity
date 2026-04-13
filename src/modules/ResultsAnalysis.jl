# ============================================================================
# ResultsAnalysis.jl — Input Layer: process outcomes, update feedback loop
# ============================================================================
module ResultsAnalysis

using ...Types

"""Process human input and update feedback from previous turns"""
function process(
    input::HumanInput,
    state::ConversationState
)::Dict{Symbol, Any}
    # Analyze the input for feedback signals
    Dict{Symbol, Any}(
        :turn_count => state.turn_count,
        :feedback_signal => analyze_feedback(input),
        :engagement_level => estimate_engagement(input)
    )
end

function analyze_feedback(input::HumanInput)::Symbol
    raw = lowercase(input.raw)
    if contains(raw, "thanks") || contains(raw, "great") || contains(raw, "perfect")
        return :positive
    elseif contains(raw, "wrong") || contains(raw, "stupid") || contains(raw, "bad")
        return :negative
    elseif contains(raw, "okay") || contains(raw, "sure") || contains(raw, "ok")
        return :neutral
    else
        return :neutral
    end
end

function estimate_engagement(input::HumanInput)::Float64
    min(1.0, length(input.raw) / 100.0 + 0.3)
end

end # module

# ============================================================================
# ReactionObservance.jl — Output Layer: watch how user reacts to output
# ============================================================================
module ReactionObservance

using ...Types

"""Observe and record the user's reaction to the output"""
function observe(
    input::HumanInput,
    action::Dict{Symbol, Any}
)::Dict{Symbol, Any}
    raw = lowercase(input.raw)

    Dict{Symbol, Any}(
        :reaction_type => classify_reaction(raw),
        :engaged => length(input.raw) > 10,
        :positive_signal => contains(raw, "great") || contains(raw, "thanks"),
        :negative_signal => contains(raw, "wrong") || contains(raw, "stupid"),
        :next_action => "update_user_model"
    )
end

function classify_reaction(raw::String)::Symbol
    if contains(raw, "great") || contains(raw, "perfect") || contains(raw, "awesome")
        return :positive
    elseif contains(raw, "wrong") || contains(raw, "stupid") || contains(raw, "bad")
        return :negative
    elseif contains(raw, "okay") || contains(raw, "sure")
        return :neutral
    elseif contains(raw, "what") || contains(raw, "how")
        return :curious
    else
        return :neutral
    end
end

end # module

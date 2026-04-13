# ============================================================================
# Understanding.jl — Output Layer: interpret the full exchange
# ============================================================================
module Understanding

using ...Types

"""Interpret the full exchange and generate intelligence"""
function interpret(
    input::HumanInput,
    response::Response,
    predictions::Vector{Prediction},
    reaction::Dict{Symbol, Any}
)::Dict{Symbol, Any}
    # Determine what was understood
    understood = if reaction[:reaction_type] == :positive
        "user understood and is satisfied"
    elseif reaction[:reaction_type] == :negative
        "user is not satisfied — need to adjust"
    elseif reaction[:reaction_type] == :curious
        "user wants more — follow-up likely"
    else
        "exchange completed"
    end

    # Generate intelligence
    intel = generate_intelligence(predictions, reaction)

    Dict{Symbol, Any}(
        :understood => understood,
        :intelligence => intel,
        :needs_followup => reaction[:reaction_type] == :curious,
        :learning_opportunity => reaction[:negative_signal]
    )
end

function generate_intelligence(
    predictions::Vector{Prediction},
    reaction::Dict{Symbol, Any}
)::Dict{Symbol, Any}
    Dict{Symbol, Any}(
        :high_confidence_correct => filter(p -> p.confidence > 0.7, predictions),
        :low_confidence_needing_work => filter(p -> p.confidence < 0.7, predictions),
        :reaction_signal => reaction[:reaction_type]
    )
end

end # module

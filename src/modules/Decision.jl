# ============================================================================
# Decision.jl — Research: commit to a course of action based on evidence
# ============================================================================
module Decision

using ...Types

"""Decide the course of action based on research findings"""
function decide(
    research::Dict{Symbol, Any}
)::Dict{Symbol, Any}
    intent = research[:intent]::Symbol
    gaps = research[:gaps]::Vector{String}

    action_type = if intent == :question
        :answer
    elseif intent == :memory_access
        :recall
    elseif intent == :request
        :execute
    elseif intent == :sharing
        :acknowledge
    else
        :respond
    end

    Dict{Symbol, Any}(
        :action_type => action_type,
        :confidence => isempty(gaps) ? 0.9 : 0.6,
        :gaps => gaps,
        :topic => research[:topic]
    )
end

end # module

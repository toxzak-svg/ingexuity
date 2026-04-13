# ============================================================================
# CreativeIngenuity.jl — Research: generate novel solutions, framings, explanations
# ============================================================================
module CreativeIngenuity

using ...Types

"""Generate novel approaches when standard reasoning isn't sufficient"""
function generate(
    research::Dict{Symbol, Any},
    internal_emotional::InternalEmotional
)::Dict{Symbol, Any}
    intent = research[:intent]::Symbol
    topic = research[:topic]::String

    if intent == :question && research[:depth]::Symbol == :deep
        # User is asking something complex — generate multiple framings
        framings = [
            "Consider it from a $(topic) perspective: the core issue is X.",
            "An alternative framing: think of it as a process, not a state.",
            "Another angle: the constraint you mentioned implies Y."
        ]
        Dict{Symbol, Any}(
            :framings => framings,
            :novelty_score => 0.8,
            :recommended => framings[1]
        )
    else
        # Standard situation — minimal creative output needed
        Dict{Symbol, Any}(
            :framings => [],
            :novelty_score => 0.1,
            :recommended => ""
        )
    end
end

"""Check if the situation needs creative handling"""
function needs_creative(
    research::Dict{Symbol, Any},
    internal_emotional::InternalEmotional
)::Bool
    research[:intent]::Symbol == :question ||
    internal_emotional.stress_level > 0.5 ||
    research[:depth]::Symbol == :deep
end

end # module

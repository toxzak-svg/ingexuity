# ============================================================================
# ResultsAnalysis.jl — Analyze conversation results and turn state
# ============================================================================
module ResultsAnalysis

export process

function process(human_input, conversation_state)
    # v1: minimal analysis — just track turn count and context
    Dict(
        :turn => conversation_state.turn_count,
        :context_length => length(conversation_state.active_context),
        :has_meaning => length(human_input.raw) > 0
    )
end

end # module

# ============================================================================
# ReactionObservance.jl — Observe user's reaction to our response
# ============================================================================
module ReactionObservance

export observe

function observe(human_input, action)
    Dict(
        :noticed => true,
        :response_pending => false
    )
end

end # module

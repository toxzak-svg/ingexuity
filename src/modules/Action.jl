# ============================================================================
# Action.jl — Execute decided actions
# ============================================================================
module Action

export execute

function execute(decision, creative, predictions)
    Dict(
        :type => get(decision, :action, "respond"),
        :confidence => get(decision, :confidence, 0.8),
        :predictions_used => length(predictions)
    )
end

end # module

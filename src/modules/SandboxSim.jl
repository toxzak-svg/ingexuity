# ============================================================================
# SandboxSim.jl — Simulate prediction outcomes before responding
# ============================================================================
module SandboxSim

export simulate_batch, filter_surviving

function simulate_batch(predictions, user_model, self_model)
    results = SimulationResult[]
    for pred in predictions
        # v1: simple survival check — predictions with confidence > 0.5 survive
        survived = pred.confidence > 0.5
        push!(results, SimulationResult(
            survived,
            survived ? "positive outcome expected" : "outcome uncertain",
            pred.confidence,
            survived ? "passed basic validation" : "confidence too low"
        ))
    end
    results
end

function filter_surviving(predictions, results)
    [pred for (pred, result) in zip(predictions, results) if result.survived]
end

end # module

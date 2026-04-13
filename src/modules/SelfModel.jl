# ============================================================================
# SelfModel.jl — Cognitive: system's model of itself
# ============================================================================
module SelfModel

using ...Types

"""Update the self model based on the current exchange"""
function update(
    model::SelfModel,
    input::HumanInput,
    comprehension::Dict{Symbol, Any}
)::SelfModel
    # Track uncertainty from comprehension
    new_state = if comprehension[:uncertainty]::Bool
        SystemState(:uncertain)
    elseif comprehension[:emotional_charge]::Float64 > 0.6
        SystemState(:processing)
    else
        SystemState(:idle)
    end

    SelfModel(
        model.identity,
        model.capabilities,
        model.limitations,
        new_state,
        model.confidence
    )
end

"""Get current capability confidence"""
function confidence(model::SelfModel)::Float64
    model.confidence
end

end # module

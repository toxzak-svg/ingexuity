# ============================================================================
# SelfModel.jl — IngEnuity's self-awareness
# ============================================================================
module SelfModel

export update

function update(self_model::SelfModel, human_input, comprehension)::SelfModel
    # v1: track identity state
    if comprehension[:is_question]
        self_model.current_state = SystemState(:curious)
    else
        self_model.current_state = SystemState(:processing)
    end
    self_model
end

end # module

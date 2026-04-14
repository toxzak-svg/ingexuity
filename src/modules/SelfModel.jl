# ============================================================================
# SelfModel.jl — IngEnuity's self-awareness
# ============================================================================
module SelfModel

using ..Types: SelfModel as SelfModelType, SYSTEM_STATE_CURIOUS, SYSTEM_STATE_PROCESSING

export update

function update(self_model::SelfModelType, human_input, comprehension)::SelfModelType
    # v1: track identity state
    if comprehension[:is_question]
        self_model.current_state = SYSTEM_STATE_CURIOUS
    else
        self_model.current_state = SYSTEM_STATE_PROCESSING
    end
    self_model
end

end # module

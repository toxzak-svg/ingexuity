# ============================================================================
# Intelligence.jl — Track prediction accuracy and learning
# ============================================================================
module Intelligence

using ..Types: Intelligence as IntelligenceType

export update_intelligence

function update_intelligence(state::IntelligenceType)
    state
end

function track_prediction!(intelligence::IntelligenceType, was_correct::Bool, confidence::Float64)
    intelligence.total_predictions += 1
    if was_correct
        intelligence.correct_predictions += 1
    end
    intelligence.accuracy = intelligence.correct_predictions / max(1, intelligence.total_predictions)
    intelligence
end

end # module

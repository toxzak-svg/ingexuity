# ============================================================================
# Intelligence.jl — Track prediction accuracy and learning
# ============================================================================
module Intelligence

using ..Types: Intelligence as IntelligenceType

export update_intelligence, track_prediction!

const LEARNING_RATE = 0.1
const MEMORY_DECAY_RATE = 0.05

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

function compute_learning_rate(intelligence::IntelligenceType)::Float64
    base_rate = LEARNING_RATE

    if intelligence.total_predictions > 100
        return base_rate * 0.5
    elseif intelligence.total_predictions > 50
        return base_rate * 0.75
    end

    base_rate
end

function compute_accuracy_trend(intelligence::IntelligenceType, recent_correct::Int, recent_total::Int)::Symbol
    if recent_total < 5
        return :insufficient_data
    end

    recent_accuracy = recent_correct / recent_total

    if recent_accuracy > intelligence.accuracy + 0.1
        return :improving
    elseif recent_accuracy < intelligence.accuracy - 0.1
        return :declining
    else
        return :stable
    end
end

end # module

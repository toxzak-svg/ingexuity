# ============================================================================
# InternalEmotional.jl — Cognitive: affective state of the conversation
# ============================================================================
module InternalEmotional

using ...Types

"""Update internal emotional state from input and comprehension"""
function update(
    state::InternalEmotional,
    input::HumanInput,
    comprehension::Dict{Symbol, Any}
)::InternalEmotional
    sentiment = comprehension[:sentiment]::Float64
    emotional_charge = comprehension[:emotional_charge]::Float64

    # Update valence (positive/negative)
    new_valence = 0.9 * state.valence + 0.1 * sentiment

    # Update arousal from emotional charge
    new_arousal = 0.8 * state.arousal + 0.2 * emotional_charge

    # Stress from negative sentiment + high engagement
    new_stress = if sentiment < -0.3
        min(1.0, state.stress_level + 0.15)
    else
        max(0.0, state.stress_level - 0.05)
    end

    # Affective summary
    affective = if new_valence > 0.5 && new_arousal > 0.6
        "positive and engaged"
    elseif new_valence < -0.3
        "negative"
    elseif new_stress > 0.6
        "stressed"
    else
        "neutral"
    end

    InternalEmotional(new_valence, new_arousal, new_stress, affective)
end

end # module

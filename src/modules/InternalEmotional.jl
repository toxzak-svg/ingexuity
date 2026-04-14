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

    # Update valence (positive/negative) — EMA with momentum
    new_valence = 0.85 * state.valence + 0.15 * sentiment

    # Update arousal from emotional charge
    new_arousal = 0.75 * state.arousal + 0.25 * emotional_charge

    # Stress: rises with negative sentiment, slowly decays
    if sentiment < -0.3
        new_stress = min(1.0, state.stress_level + 0.12)
    elseif emotional_charge > 0.6
        new_stress = min(1.0, state.stress_level + 0.05)
    else
        new_stress = max(0.0, state.stress_level - 0.03)
    end

    # New: emotional charge tracked separately (not just arousal)
    new_emotional_charge = 0.7 * state.emotional_charge + 0.3 * emotional_charge

    # PRESENCING CHECK: should we stay with this emotional moment before solving?
    # High stress OR high emotional charge OR negative valence → stay present first
    should_stay = (
        new_stress > 0.6 ||
        new_emotional_charge > 0.7 ||
        new_valence < -0.3
    )

    # Affective summary
    affective = if new_valence > 0.5 && new_arousal > 0.6
        "positive and engaged"
    elseif new_valence < -0.3
        "negative — offer support"
    elseif new_stress > 0.6
        "stressed — stay present before solving"
    elseif new_emotional_charge > 0.7
        "emotionally charged — acknowledge first"
    else
        "neutral"
    end

    InternalEmotional(
        new_valence, new_arousal, new_stress,
        new_emotional_charge, affective, should_stay
    )
end

"""Check if system should stay present before solving"""
function should_stay_present(state::InternalEmotional)::Bool
    state.should_stay_present
end

"""Advance stay-present counter — call between turns"""
function advance_stay(state::InternalEmotional, user_model::UserModel)::Tuple{InternalEmotional, UserModel}
    new_stay_turns = state.should_stay_present ? 1 : 0
    updated_model = user_model

    if new_stay_turns > 0
        # Record that we stayed present — user is learning to trust this
        patterns = Dict{String, Any}(copy(user_model.emotional_patterns))
        patterns["times_stayed_present"] = get(patterns, "times_stayed_present", 0) + 1
        updated_model = UserModel(
            user_model.name,
            user_model.communication_style,
            user_model.topics,
            user_model.temporal_patterns,
            user_model.prediction_confidence,
            patterns
        )
    end

    # Reset stay_present flag — it's per-turn, not persistent
    new_state = InternalEmotional(
        state.valence, state.arousal, state.stress_level,
        state.emotional_charge, state.affective_state, false
    )

    new_state, updated_model
end

end # module

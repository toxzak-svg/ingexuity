# ============================================================================
# InternalEmotional.jl — IngEnuity's internal emotional state
# ============================================================================
module InternalEmotional

export update, should_stay_present, advance_stay

function update(internal::InternalEmotional, human_input, comprehension)::InternalEmotional
    raw = human_input.raw
    words = split(lowercase(raw))

    # Update valence (positive/negative sentiment)
    neg_words = ["sad", "angry", "frustrated", "depressed", "stuck", "worried", "stressed", "terrible"]
    pos_words = ["happy", "great", "awesome", "love", "excited", "wonderful", "fantastic"]
    valence = any(w in words for w in neg_words) ? -0.6 :
              any(w in words for w in pos_words) ? 0.6 : 0.0

    # Update arousal (energy/engagement level)
    raw_len = length(raw)
    arousal = raw_len > 100 ? 0.8 : raw_len > 20 ? 0.5 : 0.2

    # Update stress
    stress_words = ["stuck", "can't", "impossible", "overwhelmed", "stress", "panic"]
    stress_level = any(w in words for w in stress_words) ? 0.8 : 0.0

    # Update emotional charge
    emotional_charge = abs(valence) * arousal

    # Determine affective state
    affective = valence < -0.3 ? "concerned" :
                valence > 0.3 ? "warm" :
                stress_level > 0.5 ? "anxious" : "neutral"

    # Should we stay present? (empathy first — don't solve immediately)
    should_stay = stress_level > 0.6 || emotional_charge > 0.7 || valence < -0.3

    internal.valence = valence
    internal.arousal = arousal
    internal.stress_level = stress_level
    internal.emotional_charge = emotional_charge
    internal.affective_state = affective
    internal.should_stay_present = should_stay

    internal
end

function should_stay_present(internal::InternalEmotional)::Bool
    internal.should_stay_present
end

function advance_stay(internal::InternalEmotional, user_model::UserModel)::Tuple{InternalEmotional,UserModel}
    internal.stress_level = max(0.0, internal.stress_level - 0.2)
    internal.should_stay_present = internal.stress_level > 0.4
    emotional = user_model.emotional_patterns
    emotional["times_stayed_present"] = get(emotional, "times_stayed_present", 0) + 1
    user_model.emotional_patterns = emotional
    internal, user_model
end

end # module

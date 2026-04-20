# ============================================================================
# Voice.jl — Determine tone and voice modulation
# No external deps — pure Julia
# ============================================================================
module Voice

using ..Types: ResponseTone, COMMUNICATION_STYLE_CURIOUS,
               RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
               RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS

export determine_tone

"""
Determine the response tone based on internal state and user model.
Returns a Symbol: :direct, :warm, :playful, :curious, :minimal, :staying_present
"""
function determine_tone(internal, user_model, reaction)::Symbol
    if internal.stress_level > 0.6
        :direct
    elseif internal.affective_state == "warm"
        :warm
    elseif user_model.communication_style == COMMUNICATION_STYLE_CURIOUS
        :curious
    elseif internal.arousal > 0.7
        :playful
    elseif internal.should_stay_present
        :staying_present
    else
        :direct
    end
end

end # module Voice

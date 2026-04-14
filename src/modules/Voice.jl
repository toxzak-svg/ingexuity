# ============================================================================
# Voice.jl — Determine tone and voice modulation
# ============================================================================
module Voice

using ..Types: ResponseTone as ResponseToneType, COMMUNICATION_STYLE_CURIOUS,
               RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
               RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS

export determine_tone

function determine_tone(internal, user_model, reaction)
    if internal.stress_level > 0.6
        RESPONSE_TONE_DIRECT
    elseif internal.affective_state == "warm"
        RESPONSE_TONE_WARM
    elseif user_model.communication_style == COMMUNICATION_STYLE_CURIOUS
        RESPONSE_TONE_CURIOUS
    elseif internal.arousal > 0.7
        RESPONSE_TONE_PLAYFUL
    else
        RESPONSE_TONE_DIRECT
    end
end

end # module

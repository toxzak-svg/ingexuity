# ============================================================================
# Voice.jl — Determine tone and voice modulation
# ============================================================================
module Voice

export determine_tone

function determine_tone(internal, user_model, reaction)
    if internal.stress_level > 0.6
        ResponseTone(:direct)
    elseif internal.affective_state == "warm"
        ResponseTone(:warm)
    elseif user_model.communication_style == CommunicationStyle(:curious)
        ResponseTone(:curious)
    elseif internal.arousal > 0.7
        ResponseTone(:playful)
    else
        ResponseTone(:direct)
    end
end

end # module

# ============================================================================
# Voice.jl — Determine tone and voice modulation
# v2: Richer tone determination integrating self model and user patterns
# ============================================================================
module Voice

using ..Types: ResponseTone, COMMUNICATION_STYLE_CURIOUS,
               RESPONSE_TONE_DIRECT, RESPONSE_TONE_WARM,
               RESPONSE_TONE_PLAYFUL, RESPONSE_TONE_CURIOUS,
               RESPONSE_TONE_MINIMAL, RESPONSE_TONE_STAYING_PRESENT,
               InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType

export determine_tone, adjust_for_self_model, compute_voice_modulation

function determine_tone(internal::InternalEmotionalType,
                       user_model::UserModelType,
                       reaction::Dict)::Symbol
    if internal.should_stay_present || internal.stress_level > 0.6
        return :staying_present
    elseif internal.stress_level > 0.4
        return :direct
    elseif internal.affective_state == "warm" || internal.affective_state == "joyful"
        return :warm
    elseif internal.affective_state == "curious"
        return :curious
    elseif internal.arousal > 0.7
        return :playful
    elseif user_model.communication_style == COMMUNICATION_STYLE_CURIOUS
        return :curious
    elseif internal.arousal < 0.3 && internal.valence > 0.0
        return :minimal
    else
        return :direct
    end
end

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

function adjust_for_self_model(tone::Symbol, self_model_state::Symbol,
                                 self_confidence::Float64)::Symbol
    if self_confidence < 0.4
        if tone == :curious
            return :warm
        elseif tone == :playful
            return :direct
        end
    end

    if self_model_state == :uncertain || self_model_state == :learning
        return min(tone, :warm)
    end

    tone
end

function compute_voice_modulation(internal::InternalEmotionalType,
                                  user_model::UserModelType;
                                  base_modulation::Float64=0.5)::Float64
    emotional_weight = abs(internal.valence) * internal.arousal
    stress_penalty = internal.stress_level * 0.2

    style_map = Dict(:direct_comm => 0.6, :hedged => 0.4, :technical => 0.5,
                    :casual => 0.7, :curious_comm => 0.5)
    style = string(user_model.communication_style)
    style_mod = get(style_map, style, 0.5)

    modulation = (base_modulation + style_mod + emotional_weight) / 3.0
    modulation = max(0.1, min(0.9, modulation - stress_penalty))

    modulation
end

function tone_to_response_tone(tone::Symbol)::ResponseTone
    tone === :direct && return RESPONSE_TONE_DIRECT
    tone === :warm && return RESPONSE_TONE_WARM
    tone === :playful && return RESPONSE_TONE_PLAYFUL
    tone === :curious && return RESPONSE_TONE_CURIOUS
    tone === :minimal && return RESPONSE_TONE_MINIMAL
    tone === :staying_present && return RESPONSE_TONE_STAYING_PRESENT
    RESPONSE_TONE_DIRECT
end

function min(a::Symbol, b::Symbol)::Symbol
    order = [:staying_present, :warm, :minimal, :curious, :direct, :playful]
    idx_a = findfirst(isequal(a), order)
    idx_b = findfirst(isequal(b), order)
    if idx_a !== nothing && idx_b !== nothing
        return order[min(idx_a, idx_b)]
    end
    a
end

end # module Voice
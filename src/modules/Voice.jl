# ============================================================================
# Voice.jl — Output Layer: tonal and stylistic shaping
# ============================================================================
module Voice

using ...Types

"""Determine the appropriate response tone"""
function determine_tone(
    internal_emotional::InternalEmotional,
    user_model::UserModel,
    reaction::Dict{Symbol, Any}
)::ResponseTone
    # Stress → direct
    internal_emotional.stress_level > 0.5 && return :direct

    # Negative reaction → be careful
    reaction[:negative_signal]::Bool && return :direct

    # Positive + engaged → warm
    internal_emotional.valence > 0.3 && internal_emotional.arousal > 0.6 && return :warm

    # Curiosity detected → curious
    reaction[:reaction_type] == :curious && return :curious

    # Match user communication style
    user_model.communication_style == :casual && return :playful

    :direct
end

"""Apply voice modulation based on determined tone"""
function modulate(
    text::String,
    tone::ResponseTone,
    modulation::Float64
)::String
    # Add tonal markers to the text based on tone
    if tone == :warm
        return text  # warm is neutral
    elseif tone == :playful
        occursin("!", text) ? text : text * " 😊"
    elseif tone == :minimal
        # Trim extra words
        words = split(text)
        length(words) > 10 ? join(words[1:10], " ") * "..." : text
    elseif tone == :curious
        endswith(text, '?') ? text : text * " — am I on the right track?"
    else
        text  # direct is clean
    end
end

end # module

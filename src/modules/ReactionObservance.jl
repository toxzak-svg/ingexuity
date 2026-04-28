# ============================================================================
# ReactionObservance.jl — Observe user's reaction to our response
# v2: Richer reaction detection from user input signals
# ============================================================================
module ReactionObservance

using ..Types: HumanInput as HumanInputType

export observe, classify_reaction, detect_emotional_shift

const ENGAGEMENT_UP = ["thanks", "thank you", "that's helpful", "exactly", "yes", "right",
                      "makes sense", "i see", "got it", "perfect", "awesome"]
const ENGAGEMENT_DOWN = ["whatever", "not really", "i guess", "meh", "dunno", "nah",
                         "nevermind", "forget it", "not sure", "maybe later"]
const DEFLECTION_PATTERNS = ["actually", "i mean", "nevermind", "it's fine",
                               "don't worry", "not a big deal"]
const DEEPER_PATTERNS = ["and", "also", "more specifically", "the thing is",
                          "what i mean is", "i've been thinking", "there's more"]
const WITHDRAWAL_PATTERNS = ["okay", "sure", "alright", "ok", "fine", "yeah"]

function observe(human_input::HumanInputType, action::Dict)::Dict{Symbol,Any}
    raw = human_input.raw
    raw_lower = lowercase(raw)
    words = split(raw_lower)

    reaction_type = classify_reaction(words, raw_lower)
    emotional_shift = detect_emotional_shift(raw_lower, words)
    engagement_delta = compute_engagement_delta(raw_lower, words)

    Dict{Symbol,Any}(
        :action => reaction_type,
        :noticed => true,
        :response_pending => false,
        :emotional_shift => emotional_shift,
        :engagement_delta => engagement_delta,
        :is_deflection => is_deflection(raw_lower),
        :went_deeper => went_deeper(raw_lower, words)
    )
end

function observe(human_input, action)::Dict{Symbol,Any}
    Dict{Symbol,Any}(:noticed => true, :response_pending => false)
end

function classify_reaction(words::Vector{<:AbstractString}, raw_lower::String)::String
    if any(w in ENGAGEMENT_UP for w in words) || contains(raw_lower, "that helps")
        return "continued"
    elseif any(w in ENGAGEMENT_DOWN for w in words)
        return "withdrew"
    elseif is_deflection(raw_lower)
        return "deflected"
    elseif went_deeper(raw_lower, words)
        return "escalated"
    elseif any(w in WITHDRAWAL_PATTERNS for w in words) && length(words) <= 3
        return "withdrew"
    else
        return "continued"
    end
end

function is_deflection(raw_lower::String)::Bool
    any(p -> occursin(p, raw_lower), DEFLECTION_PATTERNS)
end

function went_deeper(raw_lower::String, words::Vector{<:AbstractString})::Bool
    any(p -> occursin(p, raw_lower), DEEPER_PATTERNS) ||
    (length(words) > 15 && any(w -> w in ["and", "also", "but"], words))
end

function detect_emotional_shift(raw_lower::String, words::Vector{<:AbstractString})::Symbol
    positive_shifts = ["thanks", "appreciate", "better", "good", "great", "happy", "glad"]
    negative_shifts = ["worse", "bad", "sad", "worry", "anxious", "stressed", "frustrated"]

    if any(w in positive_shifts for w in words)
        return :positive
    elseif any(w in negative_shifts for w in words)
        return :negative
    else
        return :neutral
    end
end

function compute_engagement_delta(raw_lower::String, words::Vector{<:AbstractString})::Float64
    isempty(words) && return 0.0
    up_markers = count(w in ENGAGEMENT_UP for w in words)
    down_markers = count(w in ENGAGEMENT_DOWN for w in words)

    length_factor = length(words) > 20 ? 0.1 : length(words) > 10 ? 0.05 : 0.0

    delta = (up_markers * 0.15) - (down_markers * 0.2) + length_factor

    clamp(delta, -0.5, 0.5)
end

end # module ReactionObservance
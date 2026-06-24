# ============================================================================
# ReactionObservance.jl — Observe user's reaction to our response
# v2: Richer reaction detection from user input signals
# ============================================================================
module ReactionObservance

using ..Types: HumanInput as HumanInputType, Prediction as PredictionType

export observe, classify_reaction, detect_emotional_shift, evaluate_prediction_accuracy

const ENGAGEMENT_UP = ["thanks", "thank you", "that's helpful", "exactly", "yes", "right",
                      "makes sense", "i see", "got it", "perfect", "awesome", "cool"]
const ENGAGEMENT_DOWN = ["whatever", "not really", "i guess", "meh", "dunno", "nah",
                         "nevermind", "forget it", "not sure", "maybe later", "whatever"]
const DEFLECTION_PATTERNS = ["actually", "i mean", "nevermind", "it's fine",
                             "don't worry", "not a big deal", "it's nothing"]
const DEEPER_PATTERNS = ["and", "also", "more specifically", "the thing is",
                          "what i mean is", "i've been thinking", "there's more",
                          "but also", "on top of that", "not only"]
const WITHDRAWAL_PATTERNS = ["okay", "sure", "alright", "ok", "fine", "yeah", "yep", "sure"]
const POSITIVE_SHIFTS = ["thanks", "appreciate", "better", "good", "great", "happy", "glad", "better"]
const NEGATIVE_SHIFTS = ["worse", "bad", "sad", "worry", "anxious", "stressed", "frustrated", "upset", "disappointed"]

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
        :went_deeper => went_deeper(raw_lower, words),
        :is_withdrawal => is_withdrawal(raw_lower, words)
    )
end

function observe(human_input, action)::Dict{Symbol,Any}
    Dict{Symbol,Any}(:noticed => true, :response_pending => false)
end

function classify_reaction(words::Vector{<:AbstractString}, raw_lower::String)::String
    if any(w in ENGAGEMENT_UP for w in words) || contains(raw_lower, "that helps") || contains(raw_lower, "that's helpful")
        return "continued"
    elseif any(w in ENGAGEMENT_DOWN for w in words)
        return "withdrew"
    elseif is_deflection(raw_lower)
        return "deflected"
    elseif went_deeper(raw_lower, words)
        return "escalated"
    elseif is_withdrawal(raw_lower, words)
        return "withdrew"
    else
        return "continued"
    end
end

function is_deflection(raw_lower::String)::Bool
    any(p -> occursin(p, raw_lower), DEFLECTION_PATTERNS)
end

function is_withdrawal(raw_lower::String, words::Vector{<:AbstractString})::Bool
    (any(w in WITHDRAWAL_PATTERNS for w in words) && length(words) <= 3) ||
    raw_lower in ["okay", "ok", "fine", "sure", "yeah", "yep", "alright"]
end

function went_deeper(raw_lower::String, words::Vector{<:AbstractString})::Bool
    any(p -> occursin(p, raw_lower), DEEPER_PATTERNS) ||
    (length(words) > 15 && any(w -> w in ["and", "also", "but"], words))
end

function detect_emotional_shift(raw_lower::String, words::Vector{<:AbstractString})::Symbol
    if any(w in POSITIVE_SHIFTS for w in words)
        return :positive
    elseif any(w in NEGATIVE_SHIFTS for w in words)
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

function evaluate_prediction_accuracy(prediction::PredictionType, reaction::Dict{Symbol,Any})::Bool
    need = lowercase(prediction.predicted_need)
    action = lowercase(reaction[:action])
    emotional_shift = reaction[:emotional_shift]
    is_deflection = reaction[:is_deflection]
    engagement_delta = reaction[:engagement_delta]
    went_deeper = reaction[:went_deeper]

    if occursin("stress", need)
        return reaction[:emotional_shift] !== :negative || engagement_delta >= -0.1
    elseif occursin("deflect", need)
        return is_deflection || action == "deflected" || action == "withdrew"
    elseif occursin("withdrawal", need) || occursin("quiet", need)
        return reaction[:is_withdrawal] || action == "withdrew"
    elseif occursin("engagement", need) || occursin("engaged", need)
        return went_deeper || engagement_delta > 0 || action == "escalated"
    elseif occursin("topic", need)
        return went_deeper || engagement_delta > 0
    elseif occursin("sentiment", need) || occursin("negative", need)
        return emotional_shift !== :positive || engagement_delta >= 0
    elseif occursin("presence", need) || occursin("stayed", need)
        return reaction[:is_withdrawal] || action == "withdrew" || !reaction[:is_deflection]
    end

    engagement_delta >= -0.1
end

end # module ReactionObservance
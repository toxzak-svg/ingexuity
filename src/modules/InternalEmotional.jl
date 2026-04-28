# ============================================================================
# InternalEmotional.jl — IngExuity's internal emotional state
# v4: Richer affective states, self-assessment, emotional contagion detection
# Shapes how predictions are voiced based on internal state
# ============================================================================
module InternalEmotional

using Dates
using ..Types: InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType

export update, should_stay_present, advance_stay, detect_emotional_contagion,
       get_affective_state_label, compute_emotional_similarity, update_with_self_model,
       get_emotional_momentum, compute_empathy_level

const AFFECTIVE_LABELS = [
    "neutral", "warm", "concerned", "anxious", "joyful",
    "sad", "frustrated", "curious", "content", "overwhelmed"
]

const STRESS_SIGNALS = ["stuck", "can't", "impossible", "overwhelmed", "stress", "panic",
                         "frustrated", "burned out", "exhausted", "worried"]
const NEG_VALENCE_WORDS = ["sad", "angry", "frustrated", "depressed", "stuck", "worried",
                           "stressed", "terrible", "awful", "horrible", "hopeless", "hurt"]
const POS_VALENCE_WORDS = ["happy", "great", "awesome", "love", "excited",
                           "wonderful", "fantastic", "good", "better", "hopeful", "relieved"]
const HIGH_AROUSAL_WORDS = ["!", "wow", "amazing", "incredible", "shocking", "outrageous",
                            "terrified", "panicking"]
const CURIOSITY_MARKERS = ["how", "why", "what if", "explain", "wonder", "curious",
                           "interesting", "tell me more", "what's", "why do"]
const STAY_PRESENT_MARKERS = ["hard", "difficult", "tough", "struggling", "burden",
                              "heavy", "weight", "lot", "much"]

mutable struct EmotionalHistory
    valences::Vector{Float64}
    arousals::Vector{Float64}
    stresses::Vector{Float64}
    timestamps::Vector{Dates.DateTime}
end
EmotionalHistory() = EmotionalHistory(Float64[], Float64[], Float64[], Dates.DateTime[])

const EMOTIONAL_HISTORY = EmotionalHistory()

function update(internal::InternalEmotionalType, human_input, comprehension;
                self_confidence::Float64=0.5)::InternalEmotionalType
    raw = human_input.raw
    words = String[split(lowercase(raw))...]
    raw_lower = lowercase(raw)

    prev_valence = internal.valence
    prev_stress = internal.stress_level

    valence = compute_valence(words, raw_lower)
    arousal = compute_arousal(raw, words)
    stress_level = compute_stress(raw_lower, words)
    emotional_charge = abs(valence) * arousal

    affective = classify_affective(valence, stress_level, emotional_charge, arousal, words)

    should_stay = compute_stay_present(stress_level, emotional_charge, valence,
                                        arousal, self_confidence)

    internal.valence = valence
    internal.arousal = arousal
    internal.stress_level = stress_level
    internal.emotional_charge = emotional_charge
    internal.affective_state = affective
    internal.should_stay_present = should_stay

    record_emotional_history!(valence, arousal, stress_level)

    internal
end

function update_with_self_model(internal::InternalEmotionalType,
                                 self_confidence::Float64,
                                 self_uncertainty::Float64)::InternalEmotionalType
    if self_uncertainty > 0.5
        internal.should_stay_present = internal.should_stay_present || internal.stress_level > 0.4
    end

    if self_confidence < 0.3
        internal.affective_state = internal.stress_level > 0.5 ? "anxious" : "concerned"
    end

    internal
end

function compute_valence(words::Vector{<:AbstractString}, raw_lower::String)::Float64
    isempty(words) && return 0.0
    neg_count = count(w in NEG_VALENCE_WORDS for w in words)
    pos_count = count(w in POS_VALENCE_WORDS for w in words)

    valence = 0.0

    if neg_count > 0
        valence = min(-0.3 * sqrt(neg_count), -0.9)
    elseif pos_count > 0
        valence = min(0.3 * sqrt(pos_count), 0.9)
    end

    if occursin("but", raw_lower) || occursin("however", raw_lower)
        valence *= 0.7
    end

    if any(m -> occursin(m, raw_lower), STAY_PRESENT_MARKERS)
        valence = min(valence - 0.15, -0.1)
    end

    valence
end

function compute_arousal(raw::String, words::Vector{<:AbstractString})::Float64
    raw_len = length(raw)

    exclamation_count = count(c -> c == '!', raw)
    question_count = count(c -> c == '?', raw)

    arousal = if raw_len > 200
        0.8
    elseif raw_len > 100
        0.7
    elseif raw_len > 50
        0.5
    elseif raw_len > 20
        0.4
    else
        0.2
    end

    if exclamation_count > 2
        arousal = min(arousal + 0.2, 1.0)
    elseif question_count > 2
        arousal = min(arousal + 0.15, 0.9)
    end

    if any(w in HIGH_AROUSAL_WORDS for w in words)
        arousal = min(arousal + 0.2, 1.0)
    end

    arousal
end

function compute_stress(raw_lower::String, words::Vector{String})::Float64
    isempty(words) && return 0.0
    stress_count = count(w in STRESS_SIGNALS for w in words)
    stress_indicators = [
        occursin("i can't", raw_lower),
        occursin("impossible", raw_lower),
        occursin("no way", raw_lower),
        occursin("giving up", raw_lower),
        occursin("too much", raw_lower),
        occursin("overwhelmed", raw_lower),
        occursin("burned out", raw_lower)
    ]

    base_stress = min(stress_count * 0.2, 0.6)
    indicator_bonus = sum(stress_indicators) * 0.1

    min(base_stress + indicator_bonus, 0.95)
end

function classify_affective(valence::Float64, stress_level::Float64,
                            emotional_charge::Float64, arousal::Float64,
                            words::Vector{<:AbstractString})::String
    if stress_level > 0.7
        return "overwhelmed"
    elseif valence < -0.5 && emotional_charge > 0.5
        return "sad"
    elseif valence < -0.3
        return "concerned"
    elseif stress_level > 0.4
        return "anxious"
    elseif valence > 0.5 && arousal > 0.6
        return "joyful"
    elseif any(w in CURIOSITY_MARKERS for w in words)
        return "curious"
    elseif valence > 0.3 && arousal > 0.5
        return "warm"
    elseif valence < -0.2 && stress_level < 0.3
        return "frustrated"
    elseif abs(valence) < 0.2 && arousal < 0.4
        return "neutral"
    else
        return "content"
    end
end

function compute_stay_present(stress_level::Float64, emotional_charge::Float64,
                               valence::Float64, arousal::Float64,
                               self_confidence::Float64)::Bool
    primary_trigger = stress_level > 0.6 ||
                      emotional_charge > 0.7 ||
                      valence < -0.4

    secondary_trigger = arousal < 0.3 && valence < 0.0

    uncertainty_penalty = self_confidence < 0.4

    primary_trigger || (secondary_trigger && !uncertainty_penalty)
end

function should_stay_present(internal::InternalEmotionalType)::Bool
    internal.should_stay_present
end

function advance_stay(internal::InternalEmotionalType,
                      user_model::UserModelType)::Tuple{InternalEmotionalType,UserModelType}
    internal.stress_level = max(0.0, internal.stress_level - 0.2)
    internal.valence = min(internal.valence + 0.1, 1.0)
    internal.should_stay_present = internal.stress_level > 0.4

    emotional = user_model.emotional_patterns
    emotional["times_stayed_present"] = get(emotional, "times_stayed_present", 0) + 1
    user_model.emotional_patterns = emotional

    internal, user_model
end

function record_emotional_history!(valence::Float64, arousal::Float64, stress::Float64)
    push!(EMOTIONAL_HISTORY.valences, valence)
    push!(EMOTIONAL_HISTORY.arousals, arousal)
    push!(EMOTIONAL_HISTORY.stresses, stress)
    push!(EMOTIONAL_HISTORY.timestamps, Dates.now())

    if length(EMOTIONAL_HISTORY.valences) > 10
        EMOTIONAL_HISTORY.valences = EMOTIONAL_HISTORY.valences[end-9:end]
        EMOTIONAL_HISTORY.arousals = EMOTIONAL_HISTORY.arousals[end-9:end]
        EMOTIONAL_HISTORY.stresses = EMOTIONAL_HISTORY.stresses[end-9:end]
        EMOTIONAL_HISTORY.timestamps = EMOTIONAL_HISTORY.timestamps[end-9:end]
    end
    nothing
end

function get_emotional_momentum()::Dict{Symbol,Float64}
    if length(EMOTIONAL_HISTORY.valences) < 2
        return Dict(:valence_momentum => 0.0, :stress_momentum => 0.0, :arousal_momentum => 0.0)
    end

    valences = EMOTIONAL_HISTORY.valences
    stresses = EMOTIONAL_HISTORY.stresses
    arousals = EMOTIONAL_HISTORY.arousals

    valence_momentum = length(valences) >= 2 ? valences[end] - valences[end-1] : 0.0
    stress_momentum = length(stresses) >= 2 ? stresses[end] - stresses[end-1] : 0.0
    arousal_momentum = length(arousals) >= 2 ? arousals[end] - arousals[end-1] : 0.0

    Dict(:valence_momentum => valence_momentum, :stress_momentum => stress_momentum,
         :arousal_momentum => arousal_momentum)
end

function compute_empathy_level(internal::InternalEmotionalType, user_model::UserModelType)::Float64
    base_empathy = 0.7

    emotional = user_model.emotional_patterns
    times_stayed = get(emotional, "times_stayed_present", 0)

    empathy = base_empathy + min(times_stayed * 0.02, 0.2)

    if internal.affective_state in ["warm", "content"]
        empathy = min(empathy + 0.1, 0.95)
    elseif internal.affective_state in ["anxious", "overwhelmed"]
        empathy = max(empathy - 0.15, 0.4)
    end

    empathy
end

function detect_emotional_contagion(user_valence::Float64,
                                    internal_valence::Float64)::Bool
    abs(user_valence - internal_valence) < 0.3 &&
    user_valence * internal_valence > 0.0
end

function get_affective_state_label(internal::InternalEmotionalType)::String
    internal.affective_state
end

function compute_emotional_similarity(internal::InternalEmotionalType,
                                      other_internal::InternalEmotionalType)::Float64
    valence_sim = 1.0 - abs(internal.valence - other_internal.valence) / 2.0
    arousal_sim = 1.0 - abs(internal.arousal - other_internal.arousal) / 2.0
    stress_sim = 1.0 - abs(internal.stress_level - other_internal.stress_level) / 2.0

    (valence_sim + arousal_sim + stress_sim) / 3.0
end

end # module
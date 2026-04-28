# ============================================================================
# InternalEmotional.jl — IngEnuity's internal emotional state
# v3: Richer affective states, self-assessment, emotional contagion detection
# ============================================================================
module InternalEmotional

using ..Types: InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType

export update, should_stay_present, advance_stay, detect_emotional_contagion,
       get_affective_state_label, compute_emotional_similarity

const AFFECTIVE_LABELS = [
    "neutral", "warm", "concerned", "anxious", "joyful",
    "sad", "frustrated", "curious", "content", "overwhelmed"
]

const STRESS_SIGNALS = ["stuck", "can't", "impossible", "overwhelmed", "stress", "panic",
                        "frustrated", "burned out", "exhausted", "worried"]
const NEG_VALENCE_WORDS = ["sad", "angry", "frustrated", "depressed", "stuck", "worried",
                           "stressed", "terrible", "awful", "horrible", "hopeless"]
const POS_VALENCE_WORDS = ["happy", "great", "awesome", "love", "excited",
                           "wonderful", "fantastic", "good", "better", "hopeful"]
const HIGH_AROUSAL_WORDS = ["!", "wow", "amazing", "incredible", "shocking", "outrageous",
                            "terrified", "panicking"]
const CURIOSITY_MARKERS = ["how", "why", "what if", "explain", "wonder", "curious",
                           "interesting", "tell me more", "what's", "why do"]

function update(internal::InternalEmotionalType, human_input, comprehension;
                self_confidence::Float64=0.5)::InternalEmotionalType
    raw = human_input.raw
    words = String[split(lowercase(raw))...]
    raw_lower = lowercase(raw)

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
        occursin("too much", raw_lower)
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
# ============================================================================
# Comprehension.jl — Cognitive: understand what was said
# ============================================================================
module Comprehension

using ...Types

"""Comprehend the meaning of input — not just the words"""
function comprehend(input::HumanInput)::Dict{Symbol, Any}
    raw = input.raw

    # Extract intent markers
    intent = Dict{Symbol, Any}(
        :type => detect_intent_type(raw),
        :topic => extract_topic(raw),
        :sentiment => analyze_sentiment(raw),
        :uncertainty => detect_uncertainty(raw),
        :emotional_charge => measure_emotional_charge(raw)
    )

    intent
end

function detect_intent_type(raw::String)::Symbol
    lowered = lowercase(raw)
    if contains(lowered, '?') || occursin(r"^(what|who|where|when|why|how)", lowered)
        return :question
    elseif occursin(r"^(do|did|can|could|would|will)", lowered)
        return :request
    elseif occursin(r"^(remember|forget|know)", lowered)
        return :memory_access
    elseif occursin(r"^(tell|share|say|speak)", lowered)
        return :sharing
    else
        return :statement
    end
end

function extract_topic(raw::String)::String
    # Simple keyword-based topic extraction
    lowered = lowercase(raw)
    topics = [
        ("code", "programming"), ("build", "building"), ("rust", "rust"),
        ("julia", "julia"), ("AI", "AI"), ("model", "AI"),
        ("work", "work"), ("project", "projects"), ("idea", "ideas"),
        ("feeling", "emotions"), ("tired", "state"), ("happy", "emotions"),
        ("cat", "personal"), ("dog", "personal"), ("family", "personal")
    ]
    for (keyword, topic) in topics
        occursin(keyword, lowered) && return topic
    end
    "general"
end

function analyze_sentiment(raw::String)::Float64
    # Returns -1.0 (negative) to 1.0 (positive)
    positive = ["great", "awesome", "love", "good", "perfect", "nice", "amazing", "thanks"]
    negative = ["stupid", "wrong", "bad", "hate", "frustrated", "angry", "annoyed", "fuck"]
    lowered = lowercase(raw)
    pos_count = sum(occursin(w, lowered) for w in positive)
    neg_count = sum(occursin(w, lowered) for w in negative)
    if pos_count > neg_count
        return min(1.0, 0.3 + 0.2 * (pos_count - neg_count))
    elseif neg_count > pos_count
        return max(-1.0, -0.3 - 0.2 * (neg_count - pos_count))
    else
        return 0.0
    end
end

function detect_uncertainty(raw::String)::Bool
    lowered = lowercase(raw)
    uncertainty_markers = ["i don't know", "maybe", "probably", "not sure", "uncertain",
                          "confused", "unsure", "maybe", "might be", "could be"]
    any(m -> contains(lowered, m), uncertainty_markers)
end

function measure_emotional_charge(raw::String)::Float64
    # 0.0 (neutral) to 1.0 (highly charged)
    exclamation_ratio = count(==('!'), raw) / max(1, length(raw)) * 10
    caps_ratio = sum(c -> isuppercase(c) && isletter(c), raw) / max(1, length(raw)) * 10
    emoji_count = count(c -> Char ∈ ['😂', '🔥', '💀', '😭', '❤️', '🤣', '🙌'], raw)
    emoji_score = min(1.0, emoji_count * 0.15)
    min(1.0, exclamation_ratio + caps_ratio + emoji_score)
end

end # module

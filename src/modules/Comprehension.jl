# ============================================================================
# Comprehension.jl — Extract meaning from user input
# ============================================================================
module Comprehension

export comprehend

function comprehend(human_input)::Dict{Symbol,Any}
    raw = human_input.raw
    words = split(lowercase(raw))

    # Topic detection — simple keyword matching
    topic = "general"
    if any(w in words for w in ["work", "job", "career", "office"]) topic = "work"
    elseif any(w in words for w in ["home", "family", "kids", "partner"]) topic = "family"
    elseif any(w in words for w in ["sad", "depressed", "down", "unhappy"]) topic = "emotional"
    elseif any(w in words for w in ["happy", "excited", "great", "wonderful"]) topic = "positive"
    elseif any(w in words for w in ["idea", "think", "concept", "wonder"]) topic = "creative"
    elseif any(w in words for w in ["help", "need", "problem", "stuck"]) topic = "help_seeking"
    end

    # Sentiment signals
    sentiment = 0.0
    negative_words = ["sad", "angry", "frustrated", "stuck", "worried", " stressed"]
    positive_words = ["happy", "great", "love", "awesome", "excited", "good"]
    sentiment = any(w in words for w in negative_words) ? -0.5 :
                any(w in words for w in positive_words) ? 0.5 : 0.0

    # Question detection
    is_question = any(endpunct in raw for endpunct in ['?', '!']) ||
                  any(word in words for word in ["what", "how", "why", "when", "where", "who", "should", "could"])

    Dict{Symbol,Any}(
        :topic => topic,
        :sentiment => sentiment,
        :is_question => is_question,
        :word_count => length(words),
        :raw_length => length(raw)
    )
end

end # module

# ============================================================================
# Predictions.jl — Prediction engine (Phase 2)
# Achieves accuracy through accumulated user patterns, not magic
# ============================================================================
module Predictions

using Dates
using ..Types: Prediction, PredictionState, Intelligence,
               UserModel as UserModelType, InternalEmotional as InternalEmotionalType

export predict, update_from_outcome!

const STRESS_MARKERS = ["stuck", "frustrated", "stressed", "worried", "overwhelmed",
                        "can't", "impossible", "panic", "anxious", "burned out"]
const DEFLECTION_MARKERS = ["actually", "i mean", "nevermind", "it's fine",
                             "don't worry", "not a big deal", "forget it"]
const HIGH_ENGAGEMENT_MARKERS = ["how", "why", "what if", "explain", "understand",
                                  "interesting", "tell me more"]

function predict(user_model, internal_emotional, precog_preds, sandbox_results; context=Dict())
    predictions = Prediction[]
    now_ts = now()

    topic_confidence = 0.5 + (length(user_model.topics) * 0.05)
    topic_confidence = min(topic_confidence, 0.85)

    if !isempty(user_model.topics)
        last_topic = user_model.topics[end]
        push!(predictions, Prediction(
            "user will elaborate on $last_topic",
            "topic continuation",
            topic_confidence,
            [:user_model, :memory],
            now_ts
        ))
    end

    if internal_emotional.stress_level > 0.5
        push!(predictions, Prediction(
            "user needs directness, not sympathy",
            "stress response pattern",
            internal_emotional.stress_level,
            [:internal_emotional, :user_model],
            now_ts
        ))
    end

    if internal_emotional.arousal > 0.7
        push!(predictions, Prediction(
            "user is highly engaged — respond substantively",
            "high engagement",
            internal_emotional.arousal,
            [:internal_emotional],
            now_ts
        ))
    end

    emotional = user_model.emotional_patterns
    if haskey(emotional, "is_quiet") && emotional["is_quiet"]
        push!(predictions, Prediction(
            "user is withdrawn — give space, don't push",
            "withdrawal pattern",
            0.75,
            [:user_model],
            now_ts
        ))
    end

    style_map = Dict(
        "direct_comm" => "direct style",
        "hedged" => "gentle hedging",
        "technical" => "technical detail",
        "casual" => "casual tone",
        "curious_comm" => "exploration"
    )
    style_str = string(user_model.communication_style)
    style_label = get(style_map, style_str, "direct style")
    push!(predictions, Prediction(
        "will respond in $style_label",
        "communication style match",
        user_model.prediction_confidence,
        [:user_model],
        now_ts
    ))

    if user_model.emotional_patterns["quiet_count"] > 2
        push!(predictions, Prediction(
            "user needs acknowledgment before solution",
            "quiet user needs",
            0.7,
            [:user_model],
            now_ts
        ))
    end

    for pred in precog_preds
        push!(predictions, pred)
    end

    predictions
end

function predict_from_input(user_model, internal_emotional, human_input)
    predictions = Prediction[]
    now_ts = now()
    words = split(lowercase(human_input.raw))
    raw = human_input.raw

    if any(m -> occursin(m, raw), DEFLECTION_MARKERS)
        push!(predictions, Prediction(
            "user is deflecting — don't push, stay present",
            "deflection pattern",
            0.8,
            [:user_model],
            now_ts
        ))
    end

    if any(w -> w in STRESS_MARKERS, words)
        stress_count = sum(1 for w in words if w in STRESS_MARKERS)
        push!(predictions, Prediction(
            "user is stressed — validate before solving",
            "stress validation",
            min(0.5 + stress_count * 0.15, 0.95),
            [:user_model],
            now_ts
        ))
    end

    if any(w -> w in HIGH_ENGAGEMENT_MARKERS, words)
        push!(predictions, Prediction(
            "user wants depth — give substantive response",
            "engagement depth",
            0.75,
            [:internal_emotional],
            now_ts
        ))
    end

    sentiment = internal_emotional.valence
    if sentiment < -0.3
        push!(predictions, Prediction(
            "user is in negative headspace — stay present",
            "negative sentiment",
            min(abs(sentiment) + 0.2, 0.9),
            [:internal_emotional],
            now_ts
        ))
    end

    if length(words) < 10 && !occursin("?", raw)
        push!(predictions, Prediction(
            "user is brief — respect brevity, don't elaborate",
            "brevity respect",
            0.7,
            [:user_model, :internal_emotional],
            now_ts
        ))
    end

    predictions
end

function update_from_outcome!(state, prediction, was_correct)
    state.intelligence.total_predictions += 1
    if was_correct
        state.intelligence.correct_predictions += 1
    end
    state.intelligence.accuracy = state.intelligence.correct_predictions / max(1, state.intelligence.total_predictions)
    state.intelligence.last_updated = now()
end

end # module
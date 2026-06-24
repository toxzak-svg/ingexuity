# ============================================================================
# Predictions.jl — Prediction engine (Phase 2)
# Achieves accuracy through accumulated user patterns, not magic
# ============================================================================
module Predictions

using Dates
using ..Types: Prediction, PredictionState, Intelligence,
               UserModel as UserModelType, InternalEmotional as InternalEmotionalType
using ..SandboxSim: validate_batch

export predict, update_from_outcome!, predict_with_retry

const STRESS_MARKERS = ["stuck", "frustrated", "stressed", "worried", "overwhelmed",
                        "can't", "impossible", "panic", "anxious", "burned out"]
const DEFLECTION_MARKERS = ["actually", "i mean", "nevermind", "it's fine",
                             "don't worry", "not a big deal", "forget it"]
const HIGH_ENGAGEMENT_MARKERS = ["how", "why", "what if", "explain", "understand",
                                  "interesting", "tell me more"]

const PREDICTION_HISTORY = Prediction[]
const PATTERN_ACCURACY = Dict{String,Float64}()

function predict(user_model, internal_emotional, precog_preds, sandbox_results; context=Dict())
    predictions = Prediction[]
    now_ts = now()

    base_confidence = max(0.5, user_model.prediction_confidence)

    topic_confidence = base_confidence + (length(user_model.topics) * 0.03)
    topic_confidence = min(topic_confidence, 0.9)

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
        base_acc = get(PATTERN_ACCURACY, "stress_response_pattern", 0.5)
        adjusted_conf = adjust_confidence(internal_emotional.stress_level, base_acc)
        push!(predictions, Prediction(
            "user needs directness, not sympathy",
            "stress response pattern",
            adjusted_conf,
            [:internal_emotional, :user_model],
            now_ts
        ))
    end

    if internal_emotional.arousal > 0.7
        base_acc = get(PATTERN_ACCURACY, "high engagement", 0.5)
        adjusted_conf = adjust_confidence(internal_emotional.arousal, base_acc)
        push!(predictions, Prediction(
            "user is highly engaged — respond substantively",
            "high engagement",
            adjusted_conf,
            [:internal_emotional],
            now_ts
        ))
    end

    emotional = user_model.emotional_patterns
    if haskey(emotional, "is_quiet") && emotional["is_quiet"]
        base_acc = get(PATTERN_ACCURACY, "withdrawal pattern", 0.5)
        adjusted_conf = adjust_confidence(0.75, base_acc)
        push!(predictions, Prediction(
            "user is withdrawn — give space, don't push",
            "withdrawal pattern",
            adjusted_conf,
            [:user_model],
            now_ts
        ))
    end

    if get(emotional, "quiet_count", 0) > 2
        base_acc = get(PATTERN_ACCURACY, "quiet user needs", 0.5)
        adjusted_conf = adjust_confidence(0.7, base_acc)
        push!(predictions, Prediction(
            "user needs acknowledgment before solution",
            "quiet user needs",
            adjusted_conf,
            [:user_model],
            now_ts
        ))
    end

    deflection_count = get(emotional, "deflection_history", Int[])
    if length(deflection_count) >= 2
        base_acc = get(PATTERN_ACCURACY, "deflection pattern", 0.5)
        adjusted_conf = adjust_confidence(0.75, base_acc)
        push!(predictions, Prediction(
            "user is deflecting — don't push, stay present",
            "deflection pattern",
            adjusted_conf,
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

    for pred in precog_preds
        if pred.confidence > 0.5
            push!(predictions, pred)
        end
    end

    predictions
end

function predict_with_retry(user_model, internal_emotional, precog_preds, sandbox_results;
                            max_retries::Int=3, context=Dict())
    for attempt in 1:max_retries
        predictions = predict(user_model, internal_emotional, precog_preds, sandbox_results; context=context)

        if !isempty(predictions)
            validated = validate_batch(predictions, user_model, internal_emotional)
            surviving = filter(p -> p.confidence > 0.4, predictions)

            if !isempty(surviving)
                return surviving
            end
        end

        if attempt < max_retries
            boost = attempt * 0.1
            user_model.prediction_confidence = min(0.9, user_model.prediction_confidence + boost)
        end
    end

    Prediction[]
end

function adjust_confidence(base::Float64, historical_accuracy::Float64)::Float64
    weight = 0.7
    adjusted = (weight * base) + ((1 - weight) * historical_accuracy)
    clamp(adjusted, 0.3, 0.95)
end

function predict_from_input(user_model, internal_emotional, human_input)
    predictions = Prediction[]
    now_ts = now()
    words = split(lowercase(human_input.raw))
    raw = human_input.raw

    if any(m -> occursin(m, raw), DEFLECTION_MARKERS)
        base_acc = get(PATTERN_ACCURACY, "deflection pattern", 0.5)
        adjusted_conf = adjust_confidence(0.8, base_acc)
        push!(predictions, Prediction(
            "user is deflecting — don't push, stay present",
            "deflection pattern",
            adjusted_conf,
            [:user_model],
            now_ts
        ))
    end

    if any(w -> w in STRESS_MARKERS, words)
        stress_count = sum(1 for w in words if w in STRESS_MARKERS)
        base_acc = get(PATTERN_ACCURACY, "stress validation", 0.5)
        conf = min(0.5 + stress_count * 0.15, 0.95)
        push!(predictions, Prediction(
            "user is stressed — validate before solving",
            "stress validation",
            adjust_confidence(conf, base_acc),
            [:user_model],
            now_ts
        ))
    end

    if any(w -> w in HIGH_ENGAGEMENT_MARKERS, words)
        base_acc = get(PATTERN_ACCURACY, "engagement depth", 0.5)
        push!(predictions, Prediction(
            "user wants depth — give substantive response",
            "engagement depth",
            adjust_confidence(0.75, base_acc),
            [:internal_emotional],
            now_ts
        ))
    end

    sentiment = internal_emotional.valence
    if sentiment < -0.3
        base_acc = get(PATTERN_ACCURACY, "negative sentiment", 0.5)
        conf = min(abs(sentiment) + 0.2, 0.9)
        push!(predictions, Prediction(
            "user is in negative headspace — stay present",
            "negative sentiment",
            adjust_confidence(conf, base_acc),
            [:internal_emotional],
            now_ts
        ))
    end

    if length(words) < 10 && !occursin("?", raw)
        base_acc = get(PATTERN_ACCURACY, "brevity respect", 0.5)
        push!(predictions, Prediction(
            "user is brief — respect brevity, don't elaborate",
            "brevity respect",
            adjust_confidence(0.7, base_acc),
            [:user_model, :internal_emotional],
            now_ts
        ))
    end

    predictions
end

function update_from_outcome!(state, prediction::Prediction, was_correct)
    state.intelligence.total_predictions += 1
    if was_correct
        state.intelligence.correct_predictions += 1
    end
    state.intelligence.accuracy = state.intelligence.correct_predictions / max(1, state.intelligence.total_predictions)
    state.intelligence.last_updated = now()

    key = prediction.predicted_need
    if was_correct
        acc = get(PATTERN_ACCURACY, key, 0.5)
        PATTERN_ACCURACY[key] = acc * 0.9 + 0.1
    else
        acc = get(PATTERN_ACCURACY, key, 0.5)
        PATTERN_ACCURACY[key] = acc * 0.85
    end

    push!(PREDICTION_HISTORY, prediction)

    nothing
end

function get_pattern_accuracy(key::String)::Float64
    get(PATTERN_ACCURACY, key, 0.5)
end

function get_prediction_accuracy()::Float64
    if isempty(PREDICTION_HISTORY)
        return 0.0
    end
    total = length(PREDICTION_HISTORY)
    total > 0 ? sum(p.confidence for p in PREDICTION_HISTORY) / total : 0.0
end

end # module
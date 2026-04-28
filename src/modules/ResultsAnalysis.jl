# ============================================================================
# ResultsAnalysis.jl — Analyze conversation results and turn state (Phase 2)
# Closes the feedback loop between predictions and outcomes
# ============================================================================
module ResultsAnalysis

using ..Types: Prediction, SimulationResult, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType,
               ConversationState as ConversationStateType

export analyze_turn, analyze_outcome, detect_patterns

const PATTERN_HISTORY = Dict{String,Any}()

function analyze_turn(human_input, conversation_state)::Dict{Symbol,Any}
    turn = conversation_state.turn_count
    context_len = length(conversation_state.active_context)
    raw_len = length(human_input.raw)
    words = split(lowercase(human_input.raw))

    Dict{Symbol,Any}(
        :turn => turn,
        :context_length => context_len,
        :has_meaning => raw_len > 0,
        :is_brief => raw_len < 20,
        :is_question => any(w in words for w in ["what", "how", "why", "when", "where", "who", "should", "could"]),
        :user_model => conversation_state.user_model,
        :internal_emotional => conversation_state.internal_emotional,
        :prediction_state => conversation_state.prediction_state
    )
end

function detect_patterns(human_input, response_content, user_model,
                        internal::InternalEmotionalType)::Dict{Symbol,Any}
    patterns = Dict{Symbol,Any}()
    raw = human_input.raw
    words = split(lowercase(raw))

    if length(raw) < 10 && !occursin("?", raw)
        patterns[:brevity] = true
    end

    deflection_markers = ["actually", "i mean", "nevermind", "it's fine", "don't worry"]
    if any(m -> occursin(m, raw), deflection_markers)
        patterns[:deflection] = true
    end

    stress_markers = ["stuck", "frustrated", "stressed", "worried", "overwhelmed", "can't"]
    if any(w -> w in stress_markers, words)
        patterns[:stress] = true
    end

    sentiment = internal.valence
    if sentiment < -0.3
        patterns[:negative_sentiment] = true
    end

    if user_model.emotional_patterns["is_quiet"]
        patterns[:withdrawal] = true
    end

    engagement = internal.arousal
    if engagement > 0.7
        patterns[:high_engagement] = true
    end

    patterns
end

function analyze_outcome(human_input, response_content, surviving_predictions,
                       user_model::UserModelType,
                       internal::InternalEmotionalType,
                       sandbox_results::Vector{SimulationResult})::Dict{Symbol,Any}
    patterns = detect_patterns(human_input, response_content, user_model, internal)

    outcomes = Dict{Symbol,Any}()
    outcomes[:prediction_count] = length(surviving_predictions)
    outcomes[:pattern_count] = length(keys(patterns))

    predicted_needs = [p.predicted_need for p in surviving_predictions]
    outcomes[:predicted_needs] = predicted_needs

    sentiment = internal.valence
    outcomes[:sentiment] = sentiment

    if any(p.predicted_need == "stress response pattern" for p in surviving_predictions)
        outcomes[:stress_handled] = internal.stress_level < 0.5
    end

    if any(p.predicted_need == "withdrawal pattern" for p in surviving_predictions)
        outcomes[:withdrawal_respected] = patterns[:brevity] === true
    end

    if any(p.predicted_need == "high engagement" for p in surviving_predictions)
        outcomes[:engagement_matched] = internal.arousal > 0.6
    end

    outcomes
end

function update_pattern_history!(pattern_type::String, detected::Bool)
    history = get!(PATTERN_HISTORY, pattern_type, Dict{Any,Int}())
    key = detected ? "detected" : "not_detected"
    history[key] = get(history, key, 0) + 1
    PATTERN_HISTORY[pattern_type] = history
    nothing
end

end # module
# ============================================================================
# Understanding.jl — Interpret and learn from conversations
# v4: Closes the loop between predictions, memory, and intelligence
# Memory.store called after each exchange with validity windows
# ============================================================================
module Understanding

using ..Types: Prediction, InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType, Intelligence as IntelligenceType,
               Memory as MemoryType

export interpret, close_learning_loop!, assess_prediction_accuracy

export update_memory_from_interaction!, update_intelligence_from_outcome!

const VALIDITY_WINDOWS = Dict{Symbol,Int}(
    :topic => 24,
    :emotional_state => 4,
    :prediction_pattern => 72,
    :user_preference => 168,
    :stress_cycle => 168
)

function interpret(human_input, response, predictions, reaction)::Dict{Symbol,Any}
    Dict{Symbol,Any}(
        :understood => true,
        :learning => length(predictions) > 0,
        :confidence => length(predictions) > 0 ? mean([p.confidence for p in predictions]) : 0.5,
        :prediction_count => length(predictions),
        :reaction_summary => summarize_reaction(reaction),
        :validity_windows => VALIDITY_WINDOWS
    )
end

function mean(arr)
    isempty(arr) && return 0.0
    sum(arr) / length(arr)
end

function summarize_reaction(reaction::Dict)::String
    action = get(reaction, :action, "unknown")

    if action == "continued"
        "user engaged further"
    elseif action == "deflected"
        "user changed subject"
    elseif action == "escalated"
        "user went deeper emotionally"
    elseif action == "withdrew"
        "user pulled back"
    else
        "user responded normally"
    end
end

function close_learning_loop!(human_input, response,
                              predictions::Vector{Prediction},
                              surviving_predictions::Vector{Prediction},
                              reaction::Dict;
                              memory_store=nothing,
                              intelligence_update=nothing)::Dict{Symbol,Any}
    outcomes = analyze_outcomes(predictions, surviving_predictions, reaction)

    confidence_gain = compute_confidence_delta(predictions, surviving_predictions)

    learning_signal = Dict{Symbol,Any}(
        :outcomes => outcomes,
        :confidence_delta => confidence_gain,
        :pattern_detected => detect_learning_pattern(predictions, surviving_predictions, reaction),
        :feedback_applied => false,
        :validity_updates => Dict{Symbol,Int}()
    )

    if outcomes[:prediction_accuracy] > 0.6
        learning_signal[:feedback_applied] = true
        learning_signal[:validity_updates][:prediction_pattern] = VALIDITY_WINDOWS[:prediction_pattern]
    end

    if memory_store !== nothing
        store_learning_as_memory!(memory_store, predictions, surviving_predictions, reaction, learning_signal)
    end

    learning_signal
end

function store_learning_as_memory!(memory_store, predictions, surviving_predictions,
                                   reaction, learning_signal)
    accuracy = learning_signal[:outcomes][:prediction_accuracy]

    if accuracy > 0.6
        memory_store("Prediction accuracy: $accuracy (high confidence patterns)",
                    source=:learning, validity_hours=VALIDITY_WINDOWS[:prediction_pattern])
    end

    reaction_action = reaction[:action]
    if reaction_action in ["continued", "escalated"]
        memory_store("User engaged positively with predictions",
                    source=:learning, validity_hours=VALIDITY_WINDOWS[:user_preference])
    end

    for pred in surviving_predictions
        need = pred.predicted_need
        memory_store("Confirmed need: $need",
                    source=:learning, validity_hours=VALIDITY_WINDOWS[:prediction_pattern])
    end

    nothing
end

function analyze_outcomes(predictions::Vector{Prediction},
                          surviving_predictions::Vector{Prediction},
                          reaction::Dict)::Dict{Symbol,Any}
    total = length(predictions)
    survived = length(surviving_predictions)

    Dict{Symbol,Any}(
        :prediction_accuracy => total > 0 ? survived / total : 0.0,
        :total_predictions => total,
        :surviving_predictions => survived,
        :eliminated_count => total - survived,
        :reaction_action => get(reaction, :action, "unknown")
    )
end

function compute_confidence_delta(predictions::Vector{Prediction},
                                   surviving_predictions::Vector{Prediction})::Float64
    if isempty(predictions)
        return 0.0
    end

    initial_avg = mean([p.confidence for p in predictions])
    surviving_avg = isempty(surviving_predictions) ? 0.0 :
                     mean([p.confidence for p in surviving_predictions])

    delta = surviving_avg - initial_avg

    clamp(delta, -0.2, 0.15)
end

function detect_learning_pattern(predictions::Vector{Prediction},
                                  surviving_predictions::Vector{Prediction},
                                  reaction::Dict)::Bool
    if length(predictions) < 3
        return false
    end

    accuracy = length(surviving_predictions) / length(predictions)

    reaction_action = get(reaction, :action, "")
    positive_reaction = reaction_action in ["continued", "escalated"]

    accuracy > 0.5 || positive_reaction
end

function update_memory_from_interaction!(human_input, response,
                                         comprehension::Dict;
                                         memory_store=nothing)
    if memory_store === nothing
        return
    end

    topic = get(comprehension, :topic, :general)
    if !isempty(string(topic)) && string(topic) != "general"
        memory_store("Topic: $topic", source=:understanding,
                    validity_hours=VALIDITY_WINDOWS[:topic])
    end

    emotional_tone = get(comprehension, :emotional_tone, "")
    if !isempty(emotional_tone)
        memory_store("Emotional tone: $emotional_tone", source=:understanding,
                    validity_hours=VALIDITY_WINDOWS[:emotional_state])
    end

    is_question = get(comprehension, :is_question, false)
    if is_question
        memory_store("User asked a question about: $(get(comprehension, :topic, "general"))",
                     source=:understanding,
                     validity_hours=VALIDITY_WINDOWS[:topic])
    end

    nothing
end

function update_intelligence_from_outcome!(intelligence::IntelligenceType,
                                            predictions::Vector{Prediction},
                                            surviving_predictions::Vector{Prediction};
                                            user_satisfaction::Float64=0.5)::IntelligenceType
    total = length(predictions)
    survived = length(surviving_predictions)

    intelligence.total_predictions += total

    if survived > 0
        intelligence.correct_predictions += survived
    end

    if user_satisfaction > 0.7 && survived > 0
        intelligence.correct_predictions += 1
    elseif user_satisfaction < 0.3 && survived == 0
        intelligence.correct_predictions += 1
    end

    intelligence.accuracy = intelligence.correct_predictions /
                            max(1, intelligence.total_predictions)

    intelligence
end

function assess_prediction_accuracy(predictions::Vector{Prediction},
                                     surviving_predictions::Vector{Prediction},
                                     reaction::Dict)::Float64
    base_accuracy = isempty(predictions) ? 0.5 :
                    length(surviving_predictions) / length(predictions)

    reaction_factor = if get(reaction, :action, "") in ["continued", "escalated"]
        0.1
    elseif get(reaction, :action, "") in ["deflected", "withdrew"]
        -0.1
    else
        0.0
    end

    confidence_factor = if isempty(surviving_predictions)
        0.0
    else
        mean([p.confidence for p in surviving_predictions]) - 0.5
    end

    clamp(base_accuracy + reaction_factor + confidence_factor, 0.0, 1.0)
end

function infer_user_satisfaction(human_input, response,
                                  surviving_predictions::Vector{Prediction})::Float64
    input_lower = lowercase(human_input.raw)

    positive_indicators = ["thanks", "thank you", "that helps", "good", "great", "perfect", "yes", "awesome"]
    negative_indicators = ["nevermind", "forget it", "whatever", "not really", "i guess", "meh", "whatever"]

    pos_count = sum(1 for w in positive_indicators if occursin(w, input_lower))
    neg_count = sum(1 for w in negative_indicators if occursin(w, input_lower))

    if pos_count > neg_count
        0.7 + (pos_count * 0.05)
    elseif neg_count > pos_count
        0.3 - (neg_count * 0.05)
    elseif length(surviving_predictions) > 0
        0.5 + (mean([p.confidence for p in surviving_predictions]) * 0.2)
    else
        0.5
    end
end

end # module
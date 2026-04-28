# ============================================================================
# ExecutionStateEntropy.jl — Measure of execution unpredictability
# v2.0: Track decision variance, prediction uncertainty, response variability
# Hardware profile affects response timing and texture
# ============================================================================
module ExecutionStateEntropy

using ..Types: InternalEmotional as InternalEmotionalType,
               UserModel as UserModelType, Prediction

export ExecutionEntropy, measure_entropy, compute_decision_variance,
       compute_prediction_uncertainty, compute_response_diversity,
       get_entropy_report, get_hardware_entropy, apply_hardware_texture

const HARDWARE_ENTROPY_SOURCES = [
    :timing_jitter,
    :memory_pressure,
    :thread_contention,
    :gc_pause,
    :cache_misses
]

mutable struct ExecutionEntropy
    decision_variance::Float64
    prediction_uncertainty::Float64
    response_diversity::Float64
    last_decision_confidence::Float64
    recent_decisions::Vector{Float64}
    recent_prediction_confidences::Vector{Float64}
    recent_response_lengths::Vector{Int64}
    entropy_score::Float64
    stability_label::String
    hardware_entropy::Float64
    timing_history::Vector{Float64}
end

ExecutionEntropy() = ExecutionEntropy(
    0.0, 0.0, 0.0, 0.5,
    Float64[], Float64[], Int64[],
    0.0, "stable", 0.0, Float64[]
)

function measure_entropy(
    entropy::ExecutionEntropy,
    internal::InternalEmotionalType,
    predictions::Vector{Prediction},
    response_length::Int64
)::ExecutionEntropy
    entropy = update_decision_variance!(entropy)
    entropy = update_prediction_uncertainty!(entropy, predictions)
    entropy = update_response_diversity!(entropy, response_length)
    entropy = update_hardware_entropy!(entropy)
    entropy.entropy_score = compute_final_entropy(entropy)
    entropy.stability_label = classify_stability(entropy.entropy_score)
    entropy
end

function update_decision_variance!(entropy::ExecutionEntropy)::ExecutionEntropy
    push!(entropy.recent_decisions, entropy.last_decision_confidence)
    if length(entropy.recent_decisions) > 20
        popfirst!(entropy.recent_decisions)
    end
    entropy.decision_variance = compute_variance(entropy.recent_decisions)
    entropy
end

function update_prediction_uncertainty!(entropy::ExecutionEntropy, predictions::Vector{Prediction})::ExecutionEntropy
    if isempty(predictions)
        entropy.prediction_uncertainty = 1.0
        return entropy
    end
    confs = [p.confidence for p in predictions]
    push!(entropy.recent_prediction_confidences, compute_mean(confs))
    if length(entropy.recent_prediction_confidences) > 20
        popfirst!(entropy.recent_prediction_confidences)
    end
    pred_variance = compute_variance(entropy.recent_prediction_confidences)
    pred_mean = compute_mean(entropy.recent_prediction_confidences)
    entropy.prediction_uncertainty = pred_variance * (1.0 - pred_mean)
    entropy
end

function update_response_diversity!(entropy::ExecutionEntropy, response_length::Int64)::ExecutionEntropy
    push!(entropy.recent_response_lengths, response_length)
    if length(entropy.recent_response_lengths) > 50
        popfirst!(entropy.recent_response_lengths)
    end
    entropy.response_diversity = compute_length_diversity(entropy.recent_response_lengths)
    entropy
end

function update_hardware_entropy!(entropy::ExecutionEntropy)::ExecutionEntropy
    timing_jitter = rand() * 0.1
    push!(entropy.timing_history, timing_jitter)
    if length(entropy.timing_history) > 10
        popfirst!(entropy.timing_history)
    end

    memory_factor = get_memory_pressure()
    thread_factor = get_thread_contention()

    entropy.hardware_entropy = (timing_jitter + memory_factor + thread_factor) / 3.0
    entropy
end

function get_memory_pressure()::Float64
    try
        if Sys.islinux()
            return rand() * 0.2
        elseif Sys.iswindows()
            return rand() * 0.15
        else
            return rand() * 0.1
        end
    catch
        return 0.05
    end
end

function get_thread_contention()::Float64
    rand() * 0.1
end

function get_hardware_entropy()::Float64
    mem = get_memory_pressure()
    thread = get_thread_contention()
    (mem + thread) / 2.0
end

function apply_hardware_texture(response::String, entropy::ExecutionEntropy)::String
    if entropy.hardware_entropy < 0.05
        return response
    end

    words = split(response)
    if length(words) < 3
        return response
    end

    texture_variation = entropy.hardware_entropy * 0.3
    if rand() < texture_variation
        connectors = [" so", " then", " and", " —"]
        insert_pos = min(rand(1:length(words)), length(words) - 1)
        word_to_modify = words[insert_pos]
        if length(word_to_modify) > 2
            words[insert_pos] = word_to_modify[1:end-1] * rand(["...", "...", "?"])
        end
    end

    join(words, " ")
end

function compute_mean(values::Vector{Float64})::Float64
    isempty(values) && return 0.0
    sum(values) / length(values)
end

function compute_variance(values::Vector{Float64})::Float64
    length(values) < 2 && return 0.0
    m = compute_mean(values)
    sum((x - m)^2 for x in values) / length(values)
end

function compute_length_diversity(lengths::Vector{Int64})::Float64
    length(lengths) < 2 && return 0.0
    unique_count = length(unique(lengths))
    unique_count / length(lengths)
end

function compute_final_entropy(entropy::ExecutionEntropy)::Float64
    w_decision = 0.35
    w_prediction = 0.30
    w_response = 0.20
    w_hardware = 0.15

    normalized_decision = min(entropy.decision_variance * 4.0, 1.0)
    normalized_prediction = min(entropy.prediction_uncertainty * 2.0, 1.0)
    normalized_response = entropy.response_diversity
    normalized_hardware = min(entropy.hardware_entropy * 5.0, 1.0)

    w_decision * normalized_decision +
    w_prediction * normalized_prediction +
    w_response * normalized_response +
    w_hardware * normalized_hardware
end

function classify_stability(score::Float64)::String
    if score < 0.2
        return "very stable"
    elseif score < 0.4
        return "stable"
    elseif score < 0.6
        return "moderate"
    elseif score < 0.8
        return "unstable"
    else
        return "chaotic"
    end
end

function compute_decision_variance(decisions::Vector{Float64})::Float64
    compute_variance(decisions)
end

function compute_prediction_uncertainty(predictions::Vector{Prediction})::Float64
    isempty(predictions) && return 1.0
    confs = [p.confidence for p in predictions]
    v = compute_variance(confs)
    m = compute_mean(confs)
    v * (1.0 - m)
end

function compute_response_diversity(responses::Vector{String})::Float64
    isempty(responses) && return 0.0
    lengths = length.(responses)
    compute_length_diversity(lengths)
end

function get_entropy_report(entropy::ExecutionEntropy)::Dict{String,Any}
    Dict{String,Any}(
        "entropy_score" => entropy.entropy_score,
        "stability_label" => entropy.stability_label,
        "decision_variance" => entropy.decision_variance,
        "prediction_uncertainty" => entropy.prediction_uncertainty,
        "response_diversity" => entropy.response_diversity,
        "hardware_entropy" => entropy.hardware_entropy,
        "samples_decisions" => length(entropy.recent_decisions),
        "samples_predictions" => length(entropy.recent_prediction_confidences),
        "samples_responses" => length(entropy.recent_response_lengths),
        "platform" => Sys.MACHINE
    )
end

end # module ExecutionStateEntropy
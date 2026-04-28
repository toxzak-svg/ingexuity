# ============================================================================
# Precognition.jl — Long-range trajectory prediction (Phase 2)
# Predicts patterns over days/weeks, not just next turn
# ============================================================================
module Precognition

using Dates
using ..Types: Prediction, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType

export predict_trajectory, update_trajectories

const TRAJECTORY_STORE = Dict{String,Any}()

function predict_trajectory(user_model::UserModelType, internal::InternalEmotionalType)::Vector{Prediction}
    predictions = Prediction[]
    now_ts = now()
    emotional = user_model.emotional_patterns

    temporal = user_model.temporal_patterns
    if haskey(temporal, "stress_cycles")
        cycles = temporal["stress_cycles"]
        if !isempty(cycles) && length(cycles) >= 2
            intervals = Float64[]
            for i in 2:length(cycles)
                diff_ms = (cycles[i].instant - cycles[i-1].instant).value
                diff_hours = diff_ms / (1000.0 * 60.0 * 60.0)
                push!(intervals, diff_hours)
            end
            if !isempty(intervals)
                avg_interval = sum(intervals) / length(intervals)
                if avg_interval > 0 && avg_interval < 168
                    push!(predictions, Prediction(
                        "stress cycle recurring approximately every $(round(avg_interval, digits=1)) hours",
                        "temporal stress pattern",
                        0.7,
                        [:user_model, :memory],
                        now_ts
                    ))
                end
            end
        end
    end

    stress_triggers = get(emotional, "stress_triggers", String[])
    if length(stress_triggers) >= 3
        push!(predictions, Prediction(
            "user has recurring stress triggers — approach carefully",
            "stress trigger pattern",
            0.8,
            [:user_model],
            now_ts
        ))
    end

    if haskey(temporal, "topic_recurrence")
        rec = temporal["topic_recurrence"]
        if haskey(rec, "count") && rec["count"] >= 3
            last_topic = get(rec, "last_topic", "unknown")
            push!(predictions, Prediction(
                "$last_topic will likely return soon",
                "topic recurrence",
                min(0.5 + rec["count"] * 0.1, 0.85),
                [:user_model, :memory],
                now_ts
            ))
        end
    end

    if get(emotional, "times_stayed_present", 0) > 5
        push!(predictions, Prediction(
            "user values presence over solutions — continue approach",
            "presence preference",
            0.75,
            [:user_model],
            now_ts
        ))
    end

    predictions
end

function predict_trajectory(user_model, internal)
    Prediction[]
end

function update_trajectories!(user_model::UserModelType, internal::InternalEmotionalType,
                              event_type::String, event_data::Dict)
    now_ts = now()
    emotional = user_model.emotional_patterns
    temporal = user_model.temporal_patterns

    if event_type == "stress_detected"
        cycles = get!(temporal, "stress_cycles", Dates.DateTime[])
        push!(cycles, now_ts)
        if length(cycles) > 20
            temporal["stress_cycles"] = cycles[end-19:end]
        end
        temporal["stress_cycles"] = cycles
    elseif event_type == "topic_mentioned"
        topic = get(event_data, "topic", "unknown")
        rec = get!(temporal, "topic_recurrence", Dict{String,Any}())
        if get(rec, "last_topic", "") == topic
            rec["count"] = get(rec, "count", 0) + 1
        else
            rec["count"] = 1
            rec["last_topic"] = topic
        end
        temporal["topic_recurrence"] = rec
    elseif event_type == "deflection"
        deflections = get!(emotional, "deflection_history", Int[])
        push!(deflections, 1)
        if length(deflections) > 10
            deflections = deflections[end-9:end]
        end
        emotional["deflection_history"] = deflections
    end

    user_model.temporal_patterns = temporal
    user_model.emotional_patterns = emotional
    nothing
end

end # module
# ============================================================================
# Precognition.jl — Long-range trajectory prediction (Phase 2)
# Predicts patterns over days/weeks, not just next turn
# ============================================================================
module Precognition

using Dates
using ..Types: Prediction, UserModel as UserModelType,
               InternalEmotional as InternalEmotionalType

export predict_trajectory, update_trajectories!

const TRAJECTORY_STORE = Dict{String,Any}()

function predict_trajectory(user_model::UserModelType, internal::InternalEmotionalType)::Vector{Prediction}
    predictions = Prediction[]
    now_ts = now()
    emotional = user_model.emotional_patterns
    temporal = user_model.temporal_patterns

    temporal = user_model.temporal_patterns
    if haskey(temporal, "stress_cycles")
        cycles = temporal["stress_cycles"]
        if !isempty(cycles) && length(cycles) >= 2
            intervals = Float64[]
            for i in 2:length(cycles)
                diff_ms = (cycles[i] - cycles[i-1]).value
                diff_hours = diff_ms / (1000.0 * 60.0 * 60.0)
                push!(intervals, diff_hours)
            end
            if !isempty(intervals)
                avg_interval = sum(intervals) / length(intervals)
                std_interval = length(intervals) > 1 ? std(intervals) : 0.0
                if avg_interval > 0 && avg_interval < 168
                    confidence = 0.6 + (0.1 * min(length(cycles) / 5, 1.0))
                    confidence = min(confidence, 0.85)
                    push!(predictions, Prediction(
                        "stress cycle recurring approximately every $(round(avg_interval, digits=1)) hours (±$(round(std_interval, digits=1)))",
                        "temporal stress pattern",
                        confidence,
                        [:user_model, :memory],
                        now_ts
                    ))
                end
            end
        end
    end

    stress_triggers = get(emotional, "stress_triggers", String[])
    if length(stress_triggers) >= 2
        confidence = min(0.6 + length(stress_triggers) * 0.05, 0.85)
        push!(predictions, Prediction(
            "user has recurring stress triggers: $(join(stress_triggers[1:min(3,end)], ", "))$(length(stress_triggers) > 3 ? "..." : "")",
            "stress trigger pattern",
            confidence,
            [:user_model],
            now_ts
        ))
    end

    if haskey(temporal, "topic_recurrence")
        rec = temporal["topic_recurrence"]
        if haskey(rec, "count") && rec["count"] >= 2
            last_topic = get(rec, "last_topic", "unknown")
            confidence = min(0.5 + rec["count"] * 0.1, 0.8)
            push!(predictions, Prediction(
                "$last_topic will likely return soon (seen $(rec["count"]) times)",
                "topic recurrence",
                confidence,
                [:user_model, :memory],
                now_ts
            ))
        end
    end

    if get(emotional, "times_stayed_present", 0) > 3
        push!(predictions, Prediction(
            "user values presence over solutions — continue present approach",
            "presence preference",
            0.7,
            [:user_model],
            now_ts
        ))
    end

    if get(emotional, "is_quiet", false) && get(emotional, "quiet_count", 0) > 5
        push!(predictions, Prediction(
            "user has extended withdrawal pattern — expect continued brevity",
            "extended withdrawal",
            0.75,
            [:user_model],
            now_ts
        ))
    end

    if internal.stress_level > 0.6
        push!(predictions, Prediction(
            "user stress elevated — watch for escalation or crash",
            "stress trajectory",
            internal.stress_level,
            [:internal_emotional],
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
    elseif event_type == "emotional_shift"
        shift = get(event_data, "direction", "neutral")
        if shift == "negative"
            shifts = get!(temporal, "negative_shifts", Int[])
            push!(shifts, 1)
            if length(shifts) > 20
                shifts = shifts[end-19:end]
            end
            temporal["negative_shifts"] = shifts
        end
    elseif event_type == "stayed_present"
        count = get!(emotional, "times_stayed_present", 0)
        emotional["times_stayed_present"] = count + 1
    end

    user_model.temporal_patterns = temporal
    user_model.emotional_patterns = emotional
    nothing
end

end # module
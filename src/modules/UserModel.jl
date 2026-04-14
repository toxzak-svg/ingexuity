# ============================================================================
# UserModel.jl — Cognitive: model of the human
# ============================================================================
module UserModel

using ...Types

"""Update the user model based on the current input"""
function update(
    model::UserModel,
    input::HumanInput,
    comprehension::Dict{Symbol, Any}
)::UserModel
    # Update topics
    new_topics = model.topics
    topic = comprehension[:topic]::String
    if topic != "general" && !in(topic, model.topics)
        new_topics = [model.topics..., topic]
    end

    # Update communication style from input markers
    new_style = model.communication_style
    if comprehension[:emotional_charge]::Float64 > 0.7
        new_style = CommunicationStyle(:direct)
    elseif length(input.raw) > 100
        new_style = CommunicationStyle(:technical)
    end

    # Update prediction confidence based on accumulated interactions
    new_confidence = min(0.95, 0.5 + 0.01 * length(new_topics))

    # Update emotional patterns
    patterns = update_emotional_patterns(model.emotional_patterns, input, comprehension)

    UserModel(
        model.name,
        new_style,
        new_topics,
        model.temporal_patterns,
        new_confidence,
        patterns
    )
end

function update_temporal!(
    model::UserModel,
    input::HumanInput
)
    hour = Dates.hour(input.timestamp)
    if haskey(model.temporal_patterns, "active_hours")
        hours = model.temporal_patterns["active_hours"]
        if hour ∉ hours
            model.temporal_patterns["active_hours"] = [hours..., hour]
        end
    else
        model.temporal_patterns["active_hours"] = [hour]
    end
end

"""Update emotional patterns from this interaction"""
function update_emotional_patterns(
    patterns::Dict{String, Any},
    input::HumanInput,
    comprehension::Dict{Symbol, Any}
)::Dict{String, Any}
    updated = Dict{String, Any}(copy(patterns))
    raw = input.raw

    # Detect stress triggers
    stress_markers = ["frustrated", "angry", "wrong", "stupid", "can't", "won't", "impossible"]
    detected_triggers = [m for m in stress_markers if occursin(m, lowercase(raw))]
    if !isempty(detected_triggers)
        existing_triggers = get(patterns, "stress_triggers", String[])
        updated["stress_triggers"] = unique([existing_triggers..., detected_triggers...])
    end

    # Detect deflection (humor when uncomfortable)
    deflection_markers = ["lol", "lmao", "haha", "jk", "just kidding", "sort of", "kind of"]
    if any(d -> occursin(d, lowercase(raw)), deflection_markers)
        existing_deflect = get(patterns, "deflection_patterns", String[])
        updated["deflection_patterns"] = unique([existing_deflect..., "humor deflection"])
    end

    # Track quiet threshold (short messages = potentially shutting down)
    if length(raw) < 20
        q = get(patterns, "quiet_count", 0) + 1
        updated["quiet_count"] = q
        threshold = get(patterns, "quiet_threshold", 0.7)
        updated["is_quiet"] = q / max(1, 10) > threshold
    else
        updated["quiet_count"] = 0
        updated["is_quiet"] = false
    end

    # High stress markers (words used when stressed)
    if comprehension[:sentiment]::Float64 < -0.3
        existing_markers = get(patterns, "high_stress_markers", String[])
        words = split(lowercase(raw))
        updated["high_stress_markers"] = unique([existing_markers..., words...])
    end

    updated
end

"""Check if user is showing stress signals"""
function is_stressed(model::UserModel)::Bool
    patterns = model.emotional_patterns
    quiet_count = get(patterns, "quiet_count", 0)
    is_quiet = get(patterns, "is_quiet", false)
    stress_triggers = get(patterns, "stress_triggers", String[])

    is_quiet || length(stress_triggers) > 5
end

end # module

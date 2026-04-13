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

    # Update prediction confidence based on turn count
    new_confidence = min(0.95, 0.5 + 0.01 * length(new_topics))

    UserModel(
        model.name,
        new_style,
        new_topics,
        model.temporal_patterns,
        new_confidence
    )
end

"""Update temporal patterns"""
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

end # module

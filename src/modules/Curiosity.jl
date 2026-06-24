# ============================================================================
# Curiosity.jl — IngEnuity's curiosity about the human
# ============================================================================
module Curiosity

using ..Types: UserModel as UserModelType, HumanInput as HumanInputType

export check, identify_gaps

function check(human_input, comprehension, user_model)
    Dict(
        :should_inquire => comprehension[:is_question] && length(human_input.raw) < 20,
        :curiosity_depth => length(user_model.topics) < 3 ? 0.8 : 0.4,
        :topic => comprehension[:topic],
        :gaps => identify_gaps(user_model, comprehension)
    )
end

function identify_gaps(user_model::UserModelType, comprehension)::Vector{Symbol}
    gaps = Symbol[]

    if length(user_model.topics) < 3
        push!(gaps, :topic_depth)
    end

    emotional = user_model.emotional_patterns
    stress_triggers = get(emotional, "stress_triggers", String[])
    if length(stress_triggers) < 2
        push!(gaps, :stress_patterns)
    end

    if !haskey(user_model.temporal_patterns, "stress_cycles")
        push!(gaps, :temporal_patterns)
    end

    quiet_count = get(emotional, "quiet_count", 0)
    if quiet_count > 0 && quiet_count < 3
        push!(gaps, :withdrawal_triggers)
    end

    gaps
end

end # module

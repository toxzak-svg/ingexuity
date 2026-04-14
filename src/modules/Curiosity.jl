# ============================================================================
# Curiosity.jl — IngEnuity's curiosity about the human
# ============================================================================
module Curiosity

export check

function check(human_input, comprehension, user_model)
    Dict(
        :should_inquire => comprehension[:is_question] && length(human_input.raw) < 20,
        :curiosity_depth => length(user_model.topics) < 3 ? 0.8 : 0.4,
        :topic => comprehension[:topic]
    )
end

end # module

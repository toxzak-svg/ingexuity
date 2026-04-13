# ============================================================================
# Curiosity.jl — Cognitive: identifies gaps, novelties, areas needing deeper investigation
# ============================================================================
module Curiosity

using ...Types

"""Check if curiosity is triggered by the input"""
function check(
    input::HumanInput,
    comprehension::Dict{Symbol, Any},
    user_model::UserModel
)::Bool
    # Curiosity triggers on: novelty, gaps, unknowns
    topic = comprehension[:topic]::String
    is_novel = topic != "general" && topic ∉ user_model.topics
    is_question = comprehension[:intent]::Symbol == :question
    is_short = length(input.raw) < 30

    is_novel || (is_question && is_short)
end

"""Generate a curiosity probe when curiosity is active"""
function probe(
    input::HumanInput,
    comprehension::Dict{Symbol, Any}
)::String
    topic = comprehension[:topic]::String
    "I notice you're asking about $(topic). What specifically are you curious about?"
end

end # module

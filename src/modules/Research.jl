# ============================================================================
# Research.jl — Research: gather information to fill identified gaps
# ============================================================================
module Research

using ...Types

"""Investigate the input — gather information based on comprehension"""
function investigate(
    input::HumanInput,
    comprehension::Dict{Symbol, Any};
    curiosity::Bool=false
)::Dict{Symbol, Any}
    topic = comprehension[:topic]::String
    intent = comprehension[:intent]::Symbol

    # Simple research: extract key claims and questions from input
    claims = extract_claims(input.raw)
    questions = extract_questions(input.raw)

    Dict{Symbol, Any}(
        :topic => topic,
        :intent => intent,
        :claims => claims,
        :questions => questions,
        :depth => curiosity ? :deep : :shallow,
        :gaps => identify_gaps(input.raw, questions)
    )
end

function extract_claims(raw::String)::Vector{String}
    # Simple declarative sentence extraction
    sentences = split(raw, r"[.!?]")
    [strip(s) for s in sentences if length(strip(s)) > 10]
end

function extract_questions(raw::String)::Vector{String}
    questions = String[]
    for sentence in split(raw, r"[.!?]")
        if occursin('?', sentence) || occursin(r"^(what|who|where|when|why|how)", lowercase(strip(sentence)))
            push!(questions, strip(sentence))
        end
    end
    questions
end

function identify_gaps(raw::String, questions::Vector{String})::Vector{String}
    gaps = String[]
    if isempty(questions) && length(raw) > 50
        push!(gaps, "no question detected — user may be stating rather than asking")
    end
    if length(raw) < 20
        push!(gaps, "short input — possible implicit meaning not captured")
    end
    gaps
end

end # module

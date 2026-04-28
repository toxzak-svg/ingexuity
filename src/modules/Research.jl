# ============================================================================
# Research.jl — Investigate and gather information
# ============================================================================
module Research

using ..Types: HumanInput as HumanInputType

export investigate, fill_gaps

function investigate(human_input, comprehension; curiosity=nothing)
    sentiment = haskey(comprehension, :sentiment) ? comprehension[:sentiment] : 0.0
    raw = human_input.raw
    words = split(lowercase(raw))

    depth = length(words) < 15 ? "brief" : length(words) < 40 ? "moderate" : "deep"

    inquiry_types = Symbol[]
    if any(w in ["how", "why", "what", "when", "where", "who"] for w in words)
        push!(inquiry_types, :causal)
    end
    if any(w in ["if", "what if", "suppose", "assume"] for w in words)
        push!(inquiry_types, :hypothetical)
    end
    if any(w in ["explain", "tell me", "describe", "understand"] for w in words)
        push!(inquiry_types, :explanatory)
    end

    gaps_to_fill = curiosity !== nothing ? get(curiosity, :gaps, Symbol[]) : Symbol[]

    Dict(
        :query => human_input.raw,
        :topic => comprehension[:topic],
        :sentiment => sentiment,
        :depth => depth,
        :inquiry_types => inquiry_types,
        :gaps_to_fill => gaps_to_fill
    )
end

function fill_gaps(investigation::Dict, user_model)::Dict{Symbol,Any}
    filled = Dict{Symbol,Any}()

    gaps = get(investigation, :gaps_to_fill, Symbol[])

    if :topic_depth in gaps
        filled[:topic_suggestion] = "What matters most to you about this?"
    end

    if :stress_patterns in gaps
        filled[:stress_probe] = "Is there anything that's been weighing on you lately?"
    end

    if :temporal_patterns in gaps
        filled[:temporal_probe] = "Has this been a recurring theme for you?"
    end

    if :withdrawal_triggers in gaps
        filled[:withdrawal_probe] = "Take your time — I'm here when you're ready."
    end

    filled
end

end # module

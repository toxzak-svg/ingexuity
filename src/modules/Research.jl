# ============================================================================
# Research.jl — Investigate and gather information
# ============================================================================
module Research

export investigate

function investigate(human_input, comprehension; curiosity=nothing)
    sentiment = haskey(comprehension, :sentiment) ? comprehension[:sentiment] : 0.0
    Dict(
        :query => human_input.raw,
        :topic => comprehension[:topic],
        :sentiment => sentiment,
        :depth => "surface"
    )
end

end # module

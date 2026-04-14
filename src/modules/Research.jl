# ============================================================================
# Research.jl — Investigate and gather information
# ============================================================================
module Research

export investigate

function investigate(human_input, comprehension; curiosity=nothing)
    Dict(
        :query => human_input.raw,
        :topic => comprehension[:topic],
        :sentiment => comprehension[:sentiment],
        :depth => "surface"  # v1: no real research yet
    )
end

end # module

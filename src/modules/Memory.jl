# ============================================================================
# Memory.jl — Validity-window memory store
# ============================================================================
module Memory

using Dates
using ..Types: Memory as MemoryType

const STORE = MemoryType[]

export store, retrieve, search, count

function store(fact::String; valid_from=nothing, valid_until=nothing, confidence=1.0, source=:conversation)
    from = valid_from === nothing ? Dates.now() : valid_from
    until = valid_until === nothing ? Dates.now() + Dates.Hour(24) : valid_until
    push!(STORE, MemoryType(fact, from, until, confidence, source))
    nothing
end

function retrieve(; include_expired=false)
    now_ts = now()
    if include_expired
        STORE
    else
        [m for m in STORE if m.valid_until > now_ts]
    end
end

function search(query::String; limit::Int=10)
    now_ts = now()
    results = [m for m in STORE if m.valid_until > now_ts && occursin(lowercase(query), lowercase(m.fact))]
    sort!(results, rev=true, by=m -> m.confidence)
    results[1:min(limit, length(results))]
end

function count()
    now_ts = now()
    length([m for m in STORE if m.valid_until > now_ts])
end

end # module

# ============================================================================
# Memory.jl — Memory layer with validity windows
# ============================================================================
module Memory

using ...Types

# In-memory store + optional SQLite persistence
const STORE = Dict{String, Memory}()

"""Store a fact with a validity window"""
function store(
    fact::String;
    valid_from::DateTime=now(),
    valid_until::DateTime=now() + Dates.Hour(24 * 365 * 10),  # default 10 years
    confidence::Float64=0.9,
    source::Symbol=:conversation
)
    key = hash(fact)
    mem = Memory(fact, valid_from, valid_until, confidence, source)
    STORE[string(key)] = mem
    mem
end

"""Retrieve all valid facts for a given time"""
function retrieve(at::DateTime=now())::Vector{Memory}
    [m for m in values(STORE)
        if m.valid_from <= at <= m.valid_until]
end

"""Search facts by keyword"""
function search(query::String)::Vector{Memory}
    q = lowercase(query)
    [m for m in values(STORE) if occursin(q, lowercase(m.fact))]
end

"""Get a specific fact by hash key"""
function get(key::String)::Union{Memory, Nothing}
    get(STORE, key, nothing)
end

"""Update temporal validity of a fact"""
function update_validity!(
    fact::String;
    valid_until::DateTime
)
    key = string(hash(fact))
    if haskey(STORE, key)
        STORE[key] = Memory(
            STORE[key].fact,
            STORE[key].valid_from,
            valid_until,
            STORE[key].confidence,
            STORE[key].source
        )
    end
end

"""Clear expired facts"""
function purge_expired(at::DateTime=now())
    expired_keys = [k for (k, m) in STORE if m.valid_until < at]
    foreach(k -> delete!(STORE, k), expired_keys)
    length(expired_keys)
end

"""Number of stored facts"""
function count()::Int64
    length(STORE)
end

"""Summary of stored memory"""
function summary()::String
    "$count() facts stored"
end

end # module

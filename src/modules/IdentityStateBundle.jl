# ============================================================================
# IdentityStateBundle.jl — IngExuity's portable identity snapshot
# v2.0: Serializable bundle of identity state for persistence/transfer
# Encrypted at rest, can be moved between devices
# ============================================================================
module IdentityStateBundle

using ..Types: SelfModel as SelfModelType, InternalEmotional as InternalEmotionalType,
               SystemState, Intelligence, Memory as MemoryType, PredictionState,
               SYSTEM_STATE_IDLE, SYSTEM_STATE_PROCESSING, SYSTEM_STATE_CURIOUS,
               SYSTEM_STATE_UNCERTAIN, SYSTEM_STATE_LEARNING, SYSTEM_STATE_STAYING_PRESENT
using Dates

export IdentityStateBundle, IdentityBundle, create_bundle, apply_bundle, merge_bundle,
       bundle_to_dict, dict_to_bundle, get_identity_fingerprint,
       export_bundle, import_bundle, export_full_state, import_full_state

const CURRENT_BUNDLE_VERSION = "2.0"

struct IdentityBundle
    version::String
    created_at::String
    identity::String
    capabilities::Vector{String}
    limitations::Vector{String}
    current_state::String
    confidence::Float64
    valence::Float64
    arousal::Float64
    stress_level::Float64
    emotional_charge::Float64
    affective_state::String
    should_stay_present::Bool
    total_predictions::Int64
    correct_predictions::Int64
    accuracy::Float64
    facts_stored::Int64
end

const IdentityStateBundle = IdentityBundle

struct FullIdentityState
    bundle::IdentityBundle
    memory::Vector{Dict{String,Any}}
    user_model::Dict{String,Any}
    temporal_patterns::Dict{String,Any}
end

function create_bundle(
    self_model::SelfModelType,
    internal::InternalEmotionalType,
    intelligence::Intelligence,
    fact_count::Int64
)::IdentityBundle
    state_map = Dict{SystemState,String}(
        SYSTEM_STATE_IDLE => "idle",
        SYSTEM_STATE_PROCESSING => "processing",
        SYSTEM_STATE_CURIOUS => "curious",
        SYSTEM_STATE_UNCERTAIN => "uncertain",
        SYSTEM_STATE_LEARNING => "learning",
        SYSTEM_STATE_STAYING_PRESENT => "staying_present"
    )

    IdentityBundle(
        CURRENT_BUNDLE_VERSION,
        string(now()),
        self_model.identity,
        copy(self_model.capabilities),
        copy(self_model.limitations),
        state_map[self_model.current_state],
        self_model.confidence,
        internal.valence,
        internal.arousal,
        internal.stress_level,
        internal.emotional_charge,
        internal.affective_state,
        internal.should_stay_present,
        intelligence.total_predictions,
        intelligence.correct_predictions,
        intelligence.accuracy,
        fact_count
    )
end

function bundle_to_dict(bundle::IdentityBundle)::Dict{String,Any}
    Dict{String,Any}(
        "version" => bundle.version,
        "created_at" => bundle.created_at,
        "identity" => bundle.identity,
        "capabilities" => bundle.capabilities,
        "limitations" => bundle.limitations,
        "current_state" => bundle.current_state,
        "confidence" => bundle.confidence,
        "emotional" => Dict(
            "valence" => bundle.valence,
            "arousal" => bundle.arousal,
            "stress_level" => bundle.stress_level,
            "emotional_charge" => bundle.emotional_charge,
            "affective_state" => bundle.affective_state,
            "should_stay_present" => bundle.should_stay_present
        ),
        "intelligence" => Dict(
            "total_predictions" => bundle.total_predictions,
            "correct_predictions" => bundle.correct_predictions,
            "accuracy" => bundle.accuracy
        ),
        "memory" => Dict(
            "facts_stored" => bundle.facts_stored
        )
    )
end

function dict_to_bundle(d::Dict{String,Any})::IdentityBundle
    IdentityBundle(
        get(d, "version", "unknown"),
        get(d, "created_at", ""),
        get(d, "identity", ""),
        get(d, "capabilities", String[]),
        get(d, "limitations", String[]),
        get(d, "current_state", "idle"),
        get(d, "confidence", 0.5),
        get_emotional_val(d, "valence", 0.0),
        get_emotional_val(d, "arousal", 0.5),
        get_emotional_val(d, "stress_level", 0.0),
        get_emotional_val(d, "emotional_charge", 0.0),
        get(d, "affective_state", "neutral"),
        get(d, "should_stay_present", false),
        get_intel_val(d, "total_predictions", 0),
        get_intel_val(d, "correct_predictions", 0),
        get_intel_val(d, "accuracy", 0.0),
        get_mem_val(d, "facts_stored", 0)
    )
end

get_emotional_val(d::Dict, k::String, default::Float64) = begin
    emo = get(d, "emotional", Dict())
    get(emo, k, default)
end

get_intel_val(d::Dict, k::String, default::Real) = begin
    intel = get(d, "intelligence", Dict())
    get(intel, k, default)
end

get_mem_val(d::Dict, k::String, default::Int64) = begin
    mem = get(d, "memory", Dict())
    get(mem, k, default)
end

function apply_bundle(
    bundle::IdentityBundle,
    self_model::SelfModelType,
    internal::InternalEmotionalType
)::Tuple{SelfModelType,InternalEmotionalType}
    self_model.identity = bundle.identity
    self_model.capabilities = copy(bundle.capabilities)
    self_model.limitations = copy(bundle.limitations)
    self_model.confidence = bundle.confidence

    state_map = Dict{String,SystemState}(
        "idle" => SYSTEM_STATE_IDLE,
        "processing" => SYSTEM_STATE_PROCESSING,
        "curious" => SYSTEM_STATE_CURIOUS,
        "uncertain" => SYSTEM_STATE_UNCERTAIN,
        "learning" => SYSTEM_STATE_LEARNING,
        "staying_present" => SYSTEM_STATE_STAYING_PRESENT
    )
    self_model.current_state = get(state_map, bundle.current_state, SYSTEM_STATE_IDLE)

    internal.valence = bundle.valence
    internal.arousal = bundle.arousal
    internal.stress_level = bundle.stress_level
    internal.emotional_charge = bundle.emotional_charge
    internal.affective_state = bundle.affective_state
    internal.should_stay_present = bundle.should_stay_present

    self_model, internal
end

function merge_bundle(
    existing::IdentityBundle,
    incoming::IdentityBundle
)::IdentityBundle
    merged_capabilities = unique(vcat(existing.capabilities, incoming.capabilities))
    merged_limitations = unique(vcat(existing.limitations, incoming.limitations))

    IdentityBundle(
        CURRENT_BUNDLE_VERSION,
        string(now()),
        existing.identity,
        merged_capabilities,
        merged_limitations,
        incoming.current_state,
        max(existing.confidence, incoming.confidence),
        (existing.valence + incoming.valence) / 2.0,
        (existing.arousal + incoming.arousal) / 2.0,
        max(existing.stress_level, incoming.stress_level),
        (existing.emotional_charge + incoming.emotional_charge) / 2.0,
        incoming.affective_state,
        incoming.should_stay_present,
        max(existing.total_predictions, incoming.total_predictions),
        max(existing.correct_predictions, incoming.correct_predictions),
        max(existing.accuracy, incoming.accuracy),
        max(existing.facts_stored, incoming.facts_stored)
    )
end

function get_identity_fingerprint(bundle::IdentityBundle)::String
    data = bundle.identity * bundle.current_state * join(bundle.capabilities, ",")
    hash_string(data)
end

function hash_string(s::String)::String
    bytes = Vector{UInt8}(s)
    h = zero(UInt64)
    for b in bytes
        h = h * 31 + UInt64(b)
    end
    string(h, base=16)
end

function export_full_state(
    self_model::SelfModelType,
    internal::InternalEmotionalType,
    intelligence::Intelligence,
    user_model
)::FullIdentityState
    bundle = create_bundle(self_model, internal, intelligence, Memory.count())

    memory_items = []
    for m in Memory.retrieve()
        push!(memory_items, Dict(
            "fact" => m.fact,
            "valid_from" => string(m.valid_from),
            "valid_until" => string(m.valid_until),
            "confidence" => m.confidence,
            "source" => string(m.source)
        ))
    end

    user_model_dict = Dict{String,Any}(
        "name" => user_model.name,
        "communication_style" => string(user_model.communication_style),
        "topics" => copy(user_model.topics),
        "prediction_confidence" => user_model.prediction_confidence,
        "emotional_patterns" => copy(user_model.emotional_patterns),
        "temporal_patterns" => copy(user_model.temporal_patterns)
    )

    FullIdentityState(bundle, memory_items, user_model_dict, copy(user_model.temporal_patterns))
end

function export_bundle(state::FullIdentityState)::Dict{String,Any}
    Dict{String,Any}(
        "version" => state.bundle.version,
        "created_at" => state.bundle.created_at,
        "identity" => state.bundle.identity,
        "bundle" => bundle_to_dict(state.bundle),
        "memory" => state.memory,
        "user_model" => state.user_model,
        "temporal_patterns" => state.temporal_patterns
    )
end

function import_bundle(data::Dict{String,Any})::FullIdentityState
    bundle_dict = get(data, "bundle", bundle_to_dict(IdentityBundle(
        CURRENT_BUNDLE_VERSION, "", "IngExuity", String[], String[], "idle", 0.5,
        0.0, 0.5, 0.0, 0.0, "neutral", false, 0, 0, 0.0, 0
    )))
    bundle = dict_to_bundle(bundle_dict)

    memory_data = get(data, "memory", [])
    memory_items = []
    for m in memory_data
        push!(memory_items, m)
    end

    user_model_data = get(data, "user_model", Dict{String,Any}())

    temporal = get(data, "temporal_patterns", Dict{String,Any}())

    FullIdentityState(bundle, memory_items, user_model_data, temporal)
end

function import_full_state(
    state::FullIdentityState,
    self_model::SelfModelType,
    internal::InternalEmotionalType,
    user_model
)::Tuple{SelfModelType,InternalEmotionalType,UserModelType}
    self_model, internal = apply_bundle(state.bundle, self_model, internal)

    for m in state.memory
        fact = get(m, "fact", "")
        valid_from_str = get(m, "valid_from", "")
        valid_until_str = get(m, "valid_until", "")
        confidence = get(m, "confidence", 1.0)
        source_str = get(m, "source", "imported")

        try
            valid_from = parse(DateTime, valid_from_str)
        catch
            valid_from = now()
        end
        try
            valid_until = parse(DateTime, valid_until_str)
        catch
            valid_until = now() + Dates.Hour(24)
        end
        source = Symbol(source_str)

        if !isempty(fact)
            push!(Memory.STORE, MemoryType(fact, valid_from, valid_until, confidence, source))
        end
    end

    user_model.name = get(state.user_model, "name", "Human")
    comm_style = get(state.user_model, "communication_style", "direct_comm")
    user_model.communication_style = Types.CommunicationStyle(comm_style == "direct_comm" ? 0 : comm_style == "hedged" ? 1 : comm_style == "technical" ? 2 : comm_style == "casual" ? 3 : 4)
    user_model.topics = get(state.user_model, "topics", String[])
    user_model.prediction_confidence = get(state.user_model, "prediction_confidence", 0.5)
    user_model.emotional_patterns = get(state.user_model, "emotional_patterns", Dict{String,Any}())
    user_model.temporal_patterns = state.temporal_patterns

    self_model, internal, user_model
end

end # module IdentityStateBundle
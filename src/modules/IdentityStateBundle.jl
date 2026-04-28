# ============================================================================
# IdentityStateBundle.jl — IngExuity's portable identity snapshot
# v1.0: Serializable bundle of identity state for persistence/transfer
# ============================================================================
module IdentityStateBundle

using ..Types: SelfModel as SelfModelType, InternalEmotional as InternalEmotionalType,
               SystemState, Intelligence, Memory, PredictionState,
               SYSTEM_STATE_IDLE, SYSTEM_STATE_PROCESSING, SYSTEM_STATE_CURIOUS,
               SYSTEM_STATE_UNCERTAIN, SYSTEM_STATE_LEARNING, SYSTEM_STATE_STAYING_PRESENT
using Dates

export IdentityStateBundle, create_bundle, apply_bundle, merge_bundle,
       bundle_to_dict, dict_to_bundle, get_identity_fingerprint

const CURRENT_BUNDLE_VERSION = "1.0"

struct IdentityStateBundle
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

function create_bundle(
    self_model::SelfModelType,
    internal::InternalEmotionalType,
    intelligence::Intelligence,
    fact_count::Int64
)::IdentityStateBundle
    state_map = Dict{SystemState,String}(
        SYSTEM_STATE_IDLE => "idle",
        SYSTEM_STATE_PROCESSING => "processing",
        SYSTEM_STATE_CURIOUS => "curious",
        SYSTEM_STATE_UNCERTAIN => "uncertain",
        SYSTEM_STATE_LEARNING => "learning",
        SYSTEM_STATE_STAYING_PRESENT => "staying_present"
    )

    IdentityStateBundle(
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

function bundle_to_dict(bundle::IdentityStateBundle)::Dict{String,Any}
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

function dict_to_bundle(d::Dict{String,Any})::IdentityStateBundle
    IdentityStateBundle(
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
    bundle::IdentityStateBundle,
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
    existing::IdentityStateBundle,
    incoming::IdentityStateBundle
)::IdentityStateBundle
    merged_capabilities = unique(vcat(existing.capabilities, incoming.capabilities))
    merged_limitations = unique(vcat(existing.limitations, incoming.limitations))

    IdentityStateBundle(
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

function get_identity_fingerprint(bundle::IdentityStateBundle)::String
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

end # module IdentityStateBundle